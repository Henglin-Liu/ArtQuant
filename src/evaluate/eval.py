import argparse
import json

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

def cal_from_apddv2(total_pred_scores, total_true_scores):
    total_pred_scores = np.concatenate(total_pred_scores)
    total_true_scores = np.concatenate(total_true_scores)
    mse = mean_squared_error(total_true_scores, total_pred_scores)
    srocc, _ = spearmanr(total_true_scores, total_pred_scores)
    wasd = np.mean(np.abs(total_true_scores - total_pred_scores))

    print(f'WASD: {wasd}, MSE: {mse}, SROCC: {srocc}')

    lcc_mean = pearsonr(total_pred_scores, total_true_scores)
    print('PLCC', lcc_mean[0])

    
def parse_args():
    parser = argparse.ArgumentParser(description="evaluation parameters for DeQA-Score")
    parser.add_argument("--level_names", type=str, required=True, nargs="+")
    parser.add_argument("--pred_paths", type=str, required=True, nargs="+")
    parser.add_argument("--gt_paths", type=str, required=True, nargs="+")
    parser.add_argument("--score_weight", type=float, required=False, nargs="+",
                        default=[10.,7.5,5.,3.5,1.0]) # apddv2
    parser.add_argument("--use_openset_probs", action="store_true")
    # apddv2
    parser.add_argument("--dataset", type=str, required=False,default='baid') # apddv2
    args = parser.parse_args()
    return args



def fit_curve(x, y, curve_type="logistic_4params"):
    r"""Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m
    """
    assert curve_type in [
        "logistic_4params",
        "logistic_5params",
    ], f"curve type should be in [logistic_4params, logistic_5params], but got {curve_type}."

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.0]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(-(x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1.0 / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == "logistic_4params":
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == "logistic_5params":
        logistic = logistic_5params
        betas_init = betas_init_5params

    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    yhat = logistic(x, *betas)
    return yhat


def cal_score(level_names,score_weight, logits=None, probs=None, use_openset_probs=False,):
    if use_openset_probs:
        assert logits is None
        probs = np.array([probs[_] for _ in level_names])
    else:
        assert probs is None
        logprobs = np.array([logits[_] for _ in level_names])
        probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    score = np.inner(probs, np.array(score_weight))
    return score


if __name__ == "__main__":
    args = parse_args()
    level_names = args.level_names
    pred_paths = args.pred_paths
    gt_paths = args.gt_paths
    use_openset_probs = args.use_openset_probs

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        print("=" * 100)
        print("Pred: ", pred_path)
        print("GT: ", gt_path)

        # load predict results
        pred_metas = []
        with open(pred_path) as fr:
            for line in fr:
                pred_meta = json.loads(line)
                pred_metas.append(pred_meta)

        # load gt results
        with open(gt_path) as fr:
            gt_metas = json.load(fr)

        preds = []
        gts = []
        for pred_meta, gt_meta in zip(pred_metas, gt_metas):
            assert pred_meta["image"] == gt_meta["image"], f"image name mismatch: {pred_meta['image']} vs {gt_meta['image']}"
            if use_openset_probs:
                pred_score = cal_score(level_names,args.score_weight, probs=pred_meta["probs"], use_openset_probs=True)
            else:
                pred_score = cal_score(level_names,args.score_weight, logits=pred_meta["logits"], use_openset_probs=False)
            
            if args.dataset.lower() == 'apdd':
                max_val = 10
                min_val = 1
                lambda_bc = None 
                
            
                
            preds.append(pred_score)
            gts.append(gt_meta["gt_score"])

        preds_fit = fit_curve(preds, gts)
        for pred,gt in zip(preds, gts):
            print(pred, gt,file=open(os.path.join(os.path.split(args.pred_paths[0])[0],'pred_gt'),'a'))
        
        cal_from_apddv2([preds], [gts]) # SRCC PLCC is same as above
        

        plt.axis('equal')
        plt.scatter(gts, preds)
        plt.xlabel('GT')
        plt.ylabel('Pred')
        plt.title('GT vs Pred')
        plt.savefig(os.path.join(os.path.split(args.pred_paths[0])[0],'GT_vs_Pred.png'))
