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


