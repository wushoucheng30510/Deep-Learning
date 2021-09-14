#%%
import numpy as np
import faiss
import pandas as pd
from shopee import bert_efficientNet
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel
import torch
from tqdm import tqdm
from torch.utils.data import *
import cv2
import gc
import torchvision.transforms as transforms
from customDataset import shopeeImageDataset
from transformers import AutoModel, BertTokenizerFast

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GET_CV = True
CHECK_SUB = False

def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def read_dataset():
    if GET_CV:
        df = pd.read_csv('train.csv')
        tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
        df['matches'] = df['label_group'].map(tmp)
        df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
        if CHECK_SUB:
            df = pd.concat([df, df], axis = 0)
            df.reset_index(drop = True, inplace = True)

        image_paths = 'train_images/' + df['image']
    else:
        df = pd.read_csv('test.csv')

        image_paths = 'test_images/' + df['image']
        
    return df, image_paths


def get_test_transforms():

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((560, 560)),
            transforms.ToTensor(), # range [0, 255] -> [0.0, 0.1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

def get_embeddings(all_dataset):
    i_embeds = []
    t_embeds = []
    c_embeds = []

    image_model = EfficientNet.from_pretrained('efficientnet-b3')
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = bert_efficientNet(bert = bert, efficient_net = image_model)
    model.eval()
    
    model.load_state_dict(torch.load("best-checkpoint.pt"),strict=False)
    model = model.to(dev)

    data_loader = DataLoader(
        all_dataset,
        batch_size=16,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    with torch.no_grad():
        for img, t_seq, t_mask, _, in tqdm(data_loader):
            img = img.cuda()
            t_seq = t_seq.cuda()
            t_mask = t_mask.cuda()
            
            image, text, conca = model(image = img, sent_id = t_seq, mask = t_mask)
            image_embeddings = image.detach().cpu().numpy()
            text_embeddings = text.detach().cpu().numpy()
            conca_embeddings = conca.detach().cpu().numpy()
            i_embeds.append(image_embeddings)
            t_embeds.append(text_embeddings)
            c_embeds.append(conca_embeddings)
    
    del model
    image_embeddings = np.concatenate(i_embeds)
    text_embeddings = np.concatenate(t_embeds)
    conca_embeddings = np.concatenate(c_embeds)
    print('image embeddings shape is {}, text embeddings shape is {}, concat embeddings shape is {}'.format(image_embeddings.shape, text_embeddings.shape, conca_embeddings.shape))
    del i_embeds, t_embeds, c_embeds
    gc.collect()
    return image_embeddings, text_embeddings, conca_embeddings

#%%
def knn_fit_match_cv(df, embeddings_sample, knn, embed_option, cosine):

    num_data, num_dim = embeddings_sample.shape
    if cosine:
        model = faiss.IndexFlatIP(num_dim)
        faiss.normalize_L2(embeddings_sample)
        model.add(embeddings_sample)
        distances, indices = model.search(embeddings_sample, knn)
        distances = 1. - distances
    else:
        from sklearn.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=50, metric ='euclidean')
        model.fit(embeddings_sample)
        distances, indices = model.kneighbors(embeddings_sample)
    
    if GET_CV:
        if embed_option == 'image':
            if cosine:
                thresholds = list(np.arange(0.25,0.41,0.01)) # cosine similarity
            else:
                thresholds = list(np.arange(8,11,0.1)) # euclidean distance
        elif embed_option == 'text':
            if cosine:
                thresholds = list(np.arange(0.,0.1,0.01)) # cosine similarity
            else:
                thresholds = list(np.arange(0.,0.1,0.01)) # euclidean distance
        elif embed_option == 'concat':
            if cosine:
                thresholds = list(np.arange(0.25,0.41,0.01)) # cosine similarity
            else:
                thresholds = list(np.arange(9,12 ,0.1)) # euclidean distance
        
        scores = []
        for threshold in thresholds:
            predictions = []
            for k in range(num_data):
                idx = np.where(distances[k,] < threshold)[0]
                if len(idx) < 2:
                    idx = np.argsort(distances[0,])[:2]
                ids = indices[k,idx]
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)
            df['pred_matches'] = predictions
            df['f1'] = f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            print(f'Our f1 score for threshold {threshold} is {score}')
            scores.append(score)
            del df['f1'], df['pred_matches'], score, predictions

        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score {embed_option} is {best_score} and has a threshold {best_threshold}')
        
    del model,scores
    gc.collect()
    return distances, indices, best_threshold

def submission(df, distances, indices, threshold, embed_option):
    # Use threshold
    predictions = []
    num_data = df.shape[0]
    for k in range(num_data):
        if embed_option == 'image':
            idx = np.where(distances[k,] < threshold)[0]
        elif embed_option == 'text':
            idx = np.where(distances[k,] < threshold)[0]
        elif embed_option == 'concat':
            idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
    return predictions

def combine_predictions(row):
    x = np.concatenate([row['image_predictions_euc'], row['text_predictions_euc'], row['concate_predictions_euc'], row['image_predictions_cos'], row['text_predictions_cos'], row['concate_predictions_cos']])
    return ' '.join( np.unique(x) )
#%%
if __name__ == '__main__':
    df,image_paths = read_dataset()
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    all_dataset = shopeeImageDataset(csv_file = 'new_train.csv', root_dir = 'train_images', tokenizer = tokenizer, transform=get_test_transforms())
    
    image_embeddings, text_embeddings, concat_embeddings = get_embeddings(all_dataset)
#%%
    # # Get CV for best threshold
    # dis_img_euc, ind_img_euc, thre_img_euc = knn_fit_match_cv(df, image_embeddings, knn = 50, embed_option='image', cosine = False)
    # dis_txt_euc, ind_txt_euc, thre_txt_euc = knn_fit_match_cv(df, text_embeddings, knn = 50, embed_option='text', cosine = False)
    # dis_con_euc, ind_con_euc, thre_con_euc = knn_fit_match_cv(df, concat_embeddings, knn = 50, embed_option='concat', cosine = False)
    # dis_img_cos, ind_img_cos, thre_img_cos = knn_fit_match_cv(df, image_embeddings, knn = 50, embed_option='image', cosine = True)
    # dis_txt_cos, ind_txt_cos, thre_txt_cos = knn_fit_match_cv(df, text_embeddings, knn = 50, embed_option='text', cosine = True)
    # dis_con_cos, ind_con_cos, thre_con_cos = knn_fit_match_cv(df, concat_embeddings, knn = 50, embed_option='concat', cosine = True)
#%%
    # Caculate Euclidean distance
    dis_img_euc, ind_img_euc, thre_img_euc = knn_fit_match_cv(df, image_embeddings, knn = 50, embed_option='image', cosine = False)
    image_pred_euc = submission(df, dis_img_euc, ind_img_euc, thre_img_euc, 'image')

    dis_txt_euc, ind_txt_euc, thre_txt_euc = knn_fit_match_cv(df, text_embeddings, knn = 50, embed_option='text', cosine = False)
    txt_pred_euc = submission(df, dis_txt_euc, ind_txt_euc, thre_txt_euc, 'text')

    dis_con_euc, ind_con_euc, thre_con_euc = knn_fit_match_cv(df, concat_embeddings, knn = 50, embed_option='concat', cosine = False)
    concate_pred_euc = submission(df, dis_con_euc, ind_con_euc, thre_con_euc, 'concat')

    # Caculate cosine distances
    dis_img_cos, ind_img_cos, thre_img_cos = knn_fit_match_cv(df, image_embeddings, knn = 50, embed_option='image', cosine = True)
    image_pred_cos = submission(df, dis_img_cos, ind_img_cos, thre_img_cos, 'image')

    dis_txt_cos, ind_txt_cos, thre_txt_cos = knn_fit_match_cv(df, text_embeddings, knn = 50, embed_option='text', cosine = True)
    txt_pred_cos = submission(df, dis_txt_cos, ind_txt_cos, thre_txt_cos, 'text')

    dis_con_cos, ind_con_cos, thre_con_cos = knn_fit_match_cv(df, concat_embeddings, knn = 50, embed_option='concat', cosine = True)
    concate_pred_cos = submission(df, dis_con_cos, ind_con_cos, thre_con_cos, 'concat')

    df['image_predictions_euc'] = image_pred_euc
    df['text_predictions_euc'] = txt_pred_euc
    df['concate_predictions_euc'] = concate_pred_euc

    df['image_predictions_cos'] = image_pred_cos
    df['text_predictions_cos'] = txt_pred_cos
    df['concate_predictions_cos'] = concate_pred_cos
    
    df['pred_matches'] = df.apply(combine_predictions, axis = 1)
    df['f1'] = f1_score(df['matches'], df['pred_matches'])
    score = df['f1'].mean()
    print(f'Our final f1 cv score is {score}')
    df['matches'] = df['pred_matches']
    df[['posting_id', 'matches']].to_csv('submission.csv', index = False)
#%%