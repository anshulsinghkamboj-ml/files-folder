from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score,confusion_matrix

def model_preformance(model,X_test,y_test,preds):
    accuracy = accuracy_score(y_test, preds)

    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC: {roc}")
    print(f"PR-AUC: {pr}")

    cm = confusion_matrix(y_test, preds,normalize='true')

    tn, fp, fn, tp = cm.ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    return tn, fp, fn, tp,accuracy,roc ,pr