import pandas as pd
import numpy as np
import os, datetime, json
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from colorama import Fore
r_ = Fore.RED; sr_ = Fore.RESET

import MeCab
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from scipy.sparse import csr_matrix

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

import lightgbm as lgb

from bert_utils import *
from config import *




# ====================================== #
#                                        #
#    -- Define Settings and Constants -- #
#                                        #
# ====================================== #
settings = OrderedDict({
    "lgb_params": OrderedDict({
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "dart",
        "max_depth": -1,
        "num_leaves":40,
        "learning_rate": 0.05,
        #"bagging_freq": 5,
        #"bagging_fraction": 0.85,
        #"feature_fraction": 0.05,
        "min_data_in_leaf": 20,    # default: 20 -- 
        "max_bin": 255,            # default: 255 --
        "min_data_in_bin": 3,      # default: 3 --
        "tree_learner": "serial",  # default: "serial" --
        "boost_from_average": "false",
        "lambda_l1": 0.1,
        "lambda_l2": 30,
        "num_threads": 16,
        "verbosity": 1,
        "seed": SEED,
        }),
    "svd_components": 2048,
    "pca_components": 512,
    "rounds": 5000,
    "early_stopping_rounds": 100,
    "verbose_eval": 50,
    "folds": 5,
    "seed": SEED,
    "feature_name": [],
})



def wakati_clear(text):
    text = re.sub(r'、', '', text)
    text = re.sub(r'。', '', text)
    text = re.sub(r'\n', '', text)
    return text


def wakatier(text, tagger=MeCab.Tagger(f"-Owakati -d {dic_neologd}")):
    return wakati_clear(tagger.parse(text))


def calc_tfidf(text_list: list) -> pd.DataFrame:
    
    bow = CountVectorizer()
    tfidf = TfidfTransformer(smooth_idf=False)

    count = bow.fit_transform(text_list)
    array_bow = count.toarray() # BoW: 出現回数[dim] --
    # cf.) array_tf = array_bow / array_bow.shape[1]  # 出現確率[undim]... terms frequency, tf --
    df_tfidf = pd.DataFrame(tfidf.fit_transform(array_bow).toarray(), columns=bow.get_feature_names_out())
    df_bow = pd.DataFrame(array_bow, columns=bow.get_feature_names_out())
    return df_tfidf, df_bow


def Lgb_Metric(outputs, targets):
    return log_loss(outputs, targets)


def Lgb_train_and_predict(train, test, config, test_batch_id=None, aug=None, output_root='./output/', run_id=None, trial=False):
    
    # define run_id & make output dir --
    if not run_id:
        run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + run_id + "/"
    else:
        output_path = output_root + run_id + '/'

    # run_idが重複したらlogが消えてしまうので、プログラムごと止めるようにする --
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        if trial:
            assert True
        else:
            assert False, (f"{r_}*** ... run_id {run_id} alreadly exists ... ***{sr_}")
    
    print(f"{y_} Lgb_train_and_predict : {run_id}{sr_}")    

    # keep current source to output_path -- 
    if not os.path.exists(f"{output_path}/src"):
        os.mkdir(f"{output_path}/src")
    os.system(f'cp ./*lgb*.py {output_path}/src')
    os.system(f'cp ./*lgb*.sh {output_path}/src')
    

    # ====================================== #
    #                                        #
    #    --           Train               -- #
    #                                        #
    # ====================================== #
    oof, sub = None,None
    if train is not None:
        print(f"{g_}     ... start train{sr_}")

        # define logfile --
        log = open(output_path + '/lgb_train.log','w',buffering=1)
        log.write(str(config)+'\n')

        # define config --
        features = config['feature_name']
        params = config['lgb_params']
        rounds = config['rounds']
        verbose = config['verbose_eval']
        early_stopping_rounds = config['early_stopping_rounds']
        folds = config['folds']
        seed = SEED

        # initialize train oof & metrics --
        oof = train[[id_name]]
        oof[label_name] = 0
        all_valid_metric, feature_importance = [], []

        # make folds --
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = skf.split(train,train[label_name])
        
        # start train with folds --
        print(f"{g_}     ... folds defined{sr_}")
        for fold, (trn_index, val_index) in enumerate(split):
            print(f"{b_}        ... fold : {fold}{sr_}")
            evals_result_dic = {}
            train_cids = train.loc[trn_index,id_name].values
            
            # prepare dataset --
            if aug:
                train_aug = aug.loc[aug[id_name].isin(train_cids)]
                trn_data = lgb.Dataset(train.loc[trn_index,features].append(train_aug[features]), label=train.loc[trn_index,label_name].append(train_aug[label_name]))
            else:
                trn_data = lgb.Dataset(train.loc[trn_index,features], label=train.loc[trn_index,label_name])
            val_data = lgb.Dataset(train.loc[val_index,features], label=train.loc[val_index,label_name])
           
            # run train --
            model = lgb.train(params,
                train_set  = trn_data,
                num_boost_round   = rounds,
                valid_sets = [trn_data,val_data],
                evals_result = evals_result_dic,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose
            )
            model.save_model(output_path + '/fold%s.ckpt'%fold)

            # validation --
            valid_preds = model.predict(train.loc[val_index,features], num_iteration=model.best_iteration)
            oof.loc[val_index, label_name] = valid_preds
            for i in range(len(evals_result_dic['valid_1'][params['metric']])//verbose):
                Write_log(log,' - %i round - train_metric: %.6f - valid_metric: %.6f\n'%(i*verbose,evals_result_dic['training'][params['metric']][i*verbose],evals_result_dic['valid_1'][params['metric']][i*verbose]))
            all_valid_metric.append(Lgb_Metric(train.loc[val_index, label_name], valid_preds))
            Write_log(log,'- fold%s valid metric: %.6f\n'%(fold,all_valid_metric[-1]))

            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            feature_name = model.feature_name()
            feature_importance.append(pd.DataFrame({'feature_name':feature_name,'importance_gain':importance_gain,'importance_split':importance_split}))

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'],ascending=False)
        feature_importance_df.to_csv(output_path + '/feature_importance.csv',index=False)

        mean_valid_metric = np.mean(all_valid_metric)
        global_valid_metric = Lgb_Metric(train[label_name].values,oof[label_name].values)
        Write_log(log,'all valid mean metric:%.6f, global valid metric:%.6f'%(mean_valid_metric,global_valid_metric))

        oof.to_csv(output_path + '/oof.csv',index=False)

        log.close()

        # save log --
        log_df = pd.DataFrame({
            'run_id':[run_id],
            'mean metric':[round(mean_valid_metric,6)],
            'global metric':[round(global_valid_metric,6)],
            })
        if not os.path.exists("./experiment/lgb_experiment_log.csv"):
            log_df.to_csv("./experiment/lgb_experiment_log.csv" ,index=False)
        else:
            log_df.to_csv("./experiment/lgb_experiment_log.csv" ,index=False ,header=None ,mode='a')
        
        # save settings --
        with open(f"{output_path}/settings.json", "w") as f:
            json.dump(config, f, indent=4)


    # ====================================== #
    #                                        #
    #    --           Test                -- #
    #                                        #
    # ====================================== #
    if test is not None:
        if train is None:
            folds = config['folds']
            seed = config['seed']
            features = config['feature_name']
            mean_valid_metric, global_valid_metric = None, None

        sub = test[[id_name]]
        sub['prediction'] = 0
        for fold in tqdm(range(folds)):
            model = lgb.Booster(model_file=output_path + '/fold%s.ckpt'%fold)
            test_preds = model.predict(test[features], num_iteration=model.best_iteration)
            sub['prediction'] += (test_preds / folds)
        if test_batch_id is None:
            sub[[id_name,'prediction']].to_csv(output_path + f'/submission.csv', index=False)
        else:
            if not os.path.exists(output_path + "/test_batch"):
                os.mkdir(output_path + "/test_batch")
            sub[[id_name,'prediction']].reset_index(drop=False, names="index").to_feather(output_path + f"/test_batch/submission_batch{test_batch_id}.feather")
    
    return oof, sub, (mean_valid_metric, global_valid_metric)



# prepare data --
df, train_shape = prepare_dataframe(train_data="raw")
df["clean_text"] = df["text"].map(lambda x: clean_text(x))
text_list = df["clean_text"].values
for i in range(len(text_list)):
    text_list[i] = wakatier(text_list[i])


# tfidf -> SVD --
df_tfidf, df_bow = calc_tfidf(text_list)
df_tfidf_sparse = csr_matrix(df_tfidf)
svd = TruncatedSVD(n_components=settings["svd_components"], n_iter=30, random_state=SEED)
df_tfidf_svd = pd.DataFrame(svd.fit_transform(df_tfidf_sparse), columns=[f"svd_{str(i)}" for i in range(settings["svd_components"])])


# sentence-bert -> PCA -- 
model = SentenceTransformer("stsb-xlm-r-multilingual", device="cuda:0")
df_embeddings = model.encode(df["clean_text"].values.tolist(), convert_to_numpy=True)
pca = IncrementalPCA(n_components=settings["pca_components"])
df_bert_emb_pca = pd.DataFrame(pca.fit_transform(df_embeddings), columns=[f"pca_{str(i)}" for i in range(settings["pca_components"])])


# features --
feature_df = pd.concat([df_tfidf_svd, df_bert_emb_pca], axis=1)
settings["feature_name"] = feature_df.columns.tolist()
df = pd.concat([df, feature_df], axis=1)


# ====================================== #
#                                        #
#       -- Train-Valid & Predict --      #
#                                        #
# ====================================== #
_, _, _ = Lgb_train_and_predict(df.iloc[:train_shape-1, :], df.iloc[train_shape:, :], settings, run_id="lgb_tmp2", trial=True)
