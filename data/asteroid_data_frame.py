import pandas as pd
import numpy as np
from feature_engine.encoding import RareLabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from catboost import Pool, CatBoostClassifier


class AsteroidDataFrame:
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path)
        self.main_label = 'is_hazardeous'

    def data_frame_optim(self):
        self.df = self.df.drop_duplicates()

        self.df = self.df[self.df['pha'].isin(['Y', 'N'])]
        self.df['is_hazardeous'] = (self.df['pha'] == 'Y').astype(int)

        self.df['orbit_id'] = self.df['orbit_id'].apply(lambda x: x.replace('JPL', '').strip())

        selected_cols = ['is_hazardeous', 'neo', 'H', 'diameter', 'albedo', 'orbit_id', 'e', 'q', 'i', 'om', 'w',
       'ad', 'n', 'per', 'per_y', 'moid_ld', 'class', 'rms']
    
        self.df = self.df[selected_cols]

    def data_frame_transformation(self):

        def group_bin(x, delta=2):
            try:
                return str(delta*round(x/delta))
            except:
                return 'None'
            
        for col in ['H', 'i', 'om', 'w']:
            self.df[col] = self.df[col].apply(group_bin)
            print(f"Binned column {col}")

        def log10_bin(x):
            try:
                return str(round(1/5*round(5*np.log10(x+1)),1))
            except:
                return 'None'
            
        for col in ['diameter', 'albedo', 'e', 'q', 'ad', 'n', 'per', 'per_y', 'moid_ld', 'rms']:
            self.df[f'log10_{col}'] = self.df[col].apply(log10_bin)
            self.df = self.df.drop([col], axis=1)
            print(f"Log10-transformed column {col}")

        for col in self.df.columns:
            if col != self.main_label:
                self.df[col] = self.df[col].fillna('None').astype(str)
                encoder = RareLabelEncoder(n_categories=1, max_n_categories=150, replace_with='Other', tol=100/self.df.shape[0])
                self.df[col] = encoder.fit_transform(self.df[[col]])
                print(f"LabelEncoded column {col}")
    
    def train_data(self):
        self.data_frame_optim()
        self.data_frame_transformation()

        y = self.df[self.main_label].values.reshape(-1,)
        X = self.df.drop([self.main_label], axis=1)

        cat_cols = self.df.select_dtypes(include=['object']).columns
        cat_cols_idx = [list(X.columns).index(c) for c in cat_cols]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(class_weights)

        train_pool = Pool(X_train, 
                          y_train, 
                          cat_features=cat_cols_idx)
        test_pool = Pool(X_test,
                         y_test,
                         cat_features=cat_cols_idx)
        
        # Specify the training parameters for the CatBoostClassifier
        model = CatBoostClassifier(iterations=100,   # Number of boosting iterations
                           depth=5,          # Depth of each tree in the ensemble
                           border_count=22,  # Number of splits for numerical features
                           l2_leaf_reg=0.3,  # L2 regularization strength
                           learning_rate=2e-1, # Learning rate for gradient descent
                           class_weights=class_weights,  # Weights for class balancing
                           verbose=0)        # Control the verbosity of training
        
        model.fit(train_pool)

# Make predictions using the resulting trained model for both training and testing data
        y_train_pred = model.predict_proba(train_pool)[:, 1]  # Predicted probabilities for training data
        y_test_pred = model.predict_proba(test_pool)[:, 1]    # Predicted probabilities for testing data

# Calculate the ROC AUC score for both training and testing data
        roc_auc_train = roc_auc_score(y_train, y_train_pred)  # ROC AUC score for training data
        roc_auc_test = roc_auc_score(y_test, y_test_pred)     # ROC AUC score for testing data

        model.plot_tree(tree_idx=0)

# Print the ROC AUC scores for the training and testing data
        print(f"ROC AUC score for train {round(roc_auc_train, 6)}, and for test {round(roc_auc_test, 6)}")