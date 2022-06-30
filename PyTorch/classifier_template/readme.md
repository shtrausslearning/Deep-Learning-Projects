
'''
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400,
                  noise=0.8,
                  random_state=11)

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.2,
                                                  random_state=13,
                                                  shuffle=True)

sc = StandardScaler()
sc.fit(X_train)
'''
