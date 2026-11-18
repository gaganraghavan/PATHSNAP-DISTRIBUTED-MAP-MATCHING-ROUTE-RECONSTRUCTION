"""Trip-level model comparison for candidate-arc matching.

Loads the phase-1 feature tables, keeps only observations whose route label is
within 50 m and present in the candidate set, selects features on training trips,
then compares candidate ranking accuracy on held-out trips.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathsnap.features import MODEL_FEATURES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default='artifacts/features/train')
    ap.add_argument('--output',default='results/models')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-train-rows',type=int,default=450000)
    args=ap.parse_args(); root=Path(args.features); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    tables=[]
    for f in sorted(root.glob('*.parquet')):
        d=pd.read_parquet(f); d['trip_id']=f.stem
        # Exclude noisy route labels and candidate misses from supervised scoring.
        d=d[(d.label_reliable_50m==1)]
        if d.empty: continue
        d=d.groupby('obs',sort=False,group_keys=False).filter(lambda g: bool(g.is_label.any()))
        if not d.empty: tables.append(d)
    if len(tables)<5: raise SystemExit('Need at least five trips with reliable candidate labels.')
    data=pd.concat(tables,ignore_index=True); feature_names=list(MODEL_FEATURES)
    X=data[feature_names].replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
    y=data.is_label.astype(np.uint8).to_numpy(); groups=data.trip_id.to_numpy(); obs=data.obs.to_numpy()
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=args.seed)
    train_idx,val_idx=next(splitter.split(X,y,groups))
    rng=np.random.default_rng(args.seed)
    # Stratified row cap: retain every positive and sample negatives for fast model comparison.
    positives=train_idx[y[train_idx]==1]; negatives=train_idx[y[train_idx]==0]
    cap=max(args.max_train_rows-len(positives),len(positives))
    take=rng.choice(negatives,size=min(len(negatives),cap),replace=False)
    fit_idx=np.concatenate([positives,take]); rng.shuffle(fit_idx)
    selector_model=ExtraTreesClassifier(n_estimators=120,max_depth=16,min_samples_leaf=3,class_weight='balanced',n_jobs=-1,random_state=args.seed)
    selector_model.fit(X.iloc[fit_idx],y[fit_idx])
    selector=SelectFromModel(selector_model,threshold='median',prefit=True)
    selected=selector.get_support(); names=[n for n,s in zip(feature_names,selected) if s]
    if not names: names=feature_names; selected=np.ones(len(names),dtype=bool)
    Xtr=X.iloc[fit_idx].loc[:,names]; ytr=y[fit_idx]; Xv=X.iloc[val_idx].loc[:,names]; yv=y[val_idx]
    models={
      'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,class_weight='balanced',solver='lbfgs',random_state=args.seed)),
      'Random Forest':RandomForestClassifier(n_estimators=180,max_depth=20,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=args.seed),
      'Histogram Gradient Boosting':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=31,l2_regularization=1.0,learning_rate=0.08,random_state=args.seed),
    }
    val=data.iloc[val_idx][['trip_id','obs','is_label']].copy(); rows=[]; details=[]
    for name,model in models.items():
        model.fit(Xtr,ytr); scores=model.predict_proba(Xv)[:,1] if hasattr(model,'predict_proba') else model.decision_function(Xv)
        if name=='Logistic Regression' and scores.min()<0: scores=(scores-scores.min())/(scores.max()-scores.min()+1e-12)
        ranked=pd.DataFrame({'trip_id':groups[val_idx],'obs':obs[val_idx],'y':yv,'score':scores})
        ranked=ranked.sort_values('score',ascending=False,kind='stable')
        ranked['rank']=ranked.groupby(['trip_id','obs']).cumcount()+1
        positives_v=ranked[ranked.y==1][['trip_id','obs','rank']]
        obs_count=ranked.groupby(['trip_id','obs']).size().shape[0]
        top1=float((positives_v['rank']==1).sum()/max(obs_count,1))
        top3=float((positives_v['rank']<=3).sum()/max(obs_count,1))
        mrr=float((1/positives_v['rank']).sum()/max(obs_count,1))
        try: auc=float(roc_auc_score(yv,scores))
        except ValueError: auc=float('nan')
        try: ap_score=float(average_precision_score(yv,scores))
        except ValueError: ap_score=float('nan')
        rows.append({'model':name,'feature_count':len(names),'validation_observations':obs_count,'top1_accuracy':top1,'top3_accuracy':top3,'mean_reciprocal_rank':mrr,'candidate_roc_auc':auc,'candidate_average_precision':ap_score})
        details.append((top1,name,model))
        print(f'{name}: top1={top1:.4f}, top3={top3:.4f}, MRR={mrr:.4f}, AUC={auc:.4f}')
    results=pd.DataFrame(rows).sort_values(['top1_accuracy','top3_accuracy'],ascending=False)
    results.to_csv(out/'model_comparison.csv',index=False)
    (out/'feature_selection.json').write_text(json.dumps({'method':'ExtraTrees importance, median threshold','selected_features':names,'all_features':feature_names,'train_trip_count':len(set(groups[train_idx])),'validation_trip_count':len(set(groups[val_idx])),'validation_seed':args.seed,'max_train_rows':args.max_train_rows},indent=2))
    best=sorted(details,key=lambda z:z[0],reverse=True)[0]
    import joblib; joblib.dump({'model':best[2],'features':names,'model_name':best[1]},out/'best_model.joblib')
    print('Selected:',', '.join(names)); print('Best:',best[1])
if __name__=='__main__': main()
