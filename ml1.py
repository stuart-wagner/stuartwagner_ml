from sklearn.datasets import load_digits


digits = load_digits()      # digits ML dataset
print(digits.DESCR)         # 8x8 array entries, 
                            #    representing ink intensity. Used to guess what number it is.

print(digits.data[13])      # data for the entry
print(digits.target[13])    # data represents the target "3"

print(digits.data.shape)
print(digits.target.shape)

print(digits.images[13])



import matplotlib.pyplot as plt

figures, axes = plt.subplots(nrows=4,ncols=6,figsize=(6,4))

for item in zip(axes.ravel(), digits.images, digits.target):
    axes,image,target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
plt.show()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(digits.data, digits.target)

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11)

print(data_train.shape)
print(target_train.shape)
print(data_test.shape)
print(target_test.shape)

predicted = knn.predict(data_test)
print(predicted[:20])
expected = target_test
print(expected[:20])

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print(wrong)
print(len(wrong))


confusion = confusion_matrix(expected, predicted)
print(confusion)

