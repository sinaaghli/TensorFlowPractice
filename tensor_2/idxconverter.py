import idx2numpy
import matplotlib.pyplot as plt
ndarr = idx2numpy.convert_from_file('tmp.idx')
print(ndarr[0].shape)

b=ndarr[0].reshape(28,28)
print(b.shape)
plt.imshow(b, cmap='gray')
plt.show()