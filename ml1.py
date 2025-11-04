from sklearn.datasets import load_digits


digits = load_digits()      #digits ML dataset
print(digits.DESCR)         #8x8 array, representing ink intensity. Used to guess what number it is.

print(digits.data[13])      #data for the entry
print(digits.target[13])    #data represents the target "3"




