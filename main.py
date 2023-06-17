from preprocessing import load_dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import svm
from sklearn.metrics import classification_report as cr

# loading dataset
spams = load_dataset()

# split training and test sets
X=spams.drop(['label'], axis=1)
y=spams['label']

X = PCA(n_components=round(len(spams) * 0.08)).fit_transform(X)

X_train, X_test, y_train, y_test = tts(X,y, test_size=0.2, random_state=42)

# training to models to compare
forest = rfc()
forest.fit(X_train, y_train)

y_hat=forest.predict(X_test)

print("Random Forest Classifier")
print(forest.score(X_test,y_test))
print(cr(y_hat,y_test,zero_division=1))

support_vector_machine = svm.SVC()

support_vector_machine.fit(X_train, y_train)
y_hat=support_vector_machine.predict(X_test)

print("Support Vector Machine Classifier")
print(support_vector_machine.score(X_test,y_test))
print(cr(y_hat,y_test,zero_division=1))