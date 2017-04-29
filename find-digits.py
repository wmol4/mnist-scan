#using the neural network which is trained to center digits, scan over the large stitched image and find digits. Then use the first neural network to classify the found digits

from skimage.util.shape import view_as_windows
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import median_filter
from scipy import ndimage
from skimage.filters import gaussian

start1 = timer()
test_image = loc_X_train[130]
test_image = test_image.reshape(80, 240)
plotImage(test_image) #the original image to classify
window_shape = (28,28)

multiplier = 3

#slide by 3 to reduce the number of windows to classify
B = view_as_windows(test_image, window_shape, (multiplier, multiplier))

oldShape = B.shape
#print(oldShape)
old_shape_1 = oldShape[0]
old_shape_2 = oldShape[1]

BInput = B.reshape(-1, 784)

print("Scanning...")
print("")

tf.reset_default_graph()

with tf.Session(graph = model2) as sess:
    save2()[0].restore(sess, save2()[1])
    feed_dict = {x: BInput.reshape(old_shape_1 * old_shape_2, 28, 28, 1)}
    PREDICTIONNUMBER = np.array(prediction.eval(feed_dict))
#print(PREDICTIONNUMBER.shape)

print("Done Scanning...")
print("")

#find where the centered digits likely are:
TOTAL_WINDOW_DIMENSIONS = (old_shape_1, old_shape_2)
#predictionNumberImage = predictionNumber.reshape(TOTAL_WINDOW_DIMENSIONS)

print("Building location map...")
print("")

PREDICTIONNUMBER = np.delete(PREDICTIONNUMBER, 1, 1)
maxArray = PREDICTIONNUMBER.reshape(TOTAL_WINDOW_DIMENSIONS)

#find where pixel values are > 0.1, and blur pixels together
blobs = maxArray > .98
blobs2 = gaussian(blobs, sigma = 1)
plt.imshow(blobs2)
plt.show()

middle1 = timer()
print("Time taken scanning and locating digits:", middle1 - start1)

#label connected regions
labels, nlabels = ndimage.label(blobs2)

#find their centers of mass, weighted by pixel values (more weight for a higher probability of being "centered")
#r = row, c = column
r, c = np.vstack(ndimage.center_of_mass(maxArray, labels, np.arange(nlabels) + 1)).T
row = (r*multiplier).T
column = (c*multiplier).T

#reorder the arrays so they go left to right
column_ordered, row_ordered = zip(*sorted(zip(column, row)))
column_ordered = list(column_ordered)
row_ordered = list(row_ordered)

NUMBER_OF_DIGITS = len(row_ordered)

print(NUMBER_OF_DIGITS, "were found.")

#round to the nearest integer
digits = []

print("The digits that were found:")
print("")

for i in range(NUMBER_OF_DIGITS):
    row_ordered[i] = int(row_ordered[i])
    column_ordered[i] = int(column_ordered[i])
    cropped = test_image[row_ordered[i]:row_ordered[i]+28, column_ordered[i]: column_ordered[i] + 28]
    plotImage(cropped)
    digits.append(cropped)
digits = np.array(digits)

tf.reset_default_graph()
with tf.Session(graph = model1) as sess:
    save1()[0].restore(sess, save1()[1])
    feed_dict = {x_1: digits.reshape(NUMBER_OF_DIGITS, 28, 28, 1), keep_prob: 1.}
    predictions = np.array(absoluteAnswerLarge(layer_5_1.eval(feed_dict)))


print("")
print("The final predictions")
predictions = np.array(predictions)
print(predictions)

print("")
print("The truth values")
print(full_y_train[130].reshape(5, 11))

end1 = timer()
print("Time spent on one image:", end1 - start1)
