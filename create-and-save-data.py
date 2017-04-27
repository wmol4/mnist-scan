#this code is to create MNIST images that are all stitched together. See the helper functions file for the functions used in this code.

#import data
trainX = mnist.train.images
trainX = trainX.reshape(55000,784)

trainY = mnist.train.labels

testX = mnist.test.images
testX = testX.reshape(10000, 784)

testY = mnist.test.labels

#build a DIGIT / NO DIGIT dataset
#Digit: MNIST images
#NO DIGIT: black squares, offset MNIST images
def buildOffset(trainX):
    is_it_a_digit = np.array([range(784)])
    is_it_a_digit_label = np.array([range(2)])

    for i in range(9000):
        if i%90 == 0:
            print(str(i / 90), "percent complete. ", sep = ' ', end='', flush=True)

        if i in range(0,500):
            zero_array = np.zeros((28,28))
            zero_array = zero_array.ravel()
            is_it_a_digit = np.concatenate((is_it_a_digit, np.array([zero_array])), axis = 0)
            is_it_a_digit_label = np.concatenate((is_it_a_digit_label, np.array([[0, 1]])), axis = 0) #0 for black space

        elif i in range(500, 6000):
                    #take a random MNIST image:
            # - if up, add randomUp 0's to the bottom, remove randomUp rows from the top
            # - if down, add randomDown 0's to the top, remove randomDown rows from the bottom
            # - if left, add randomLeft 0's to the right, remove randomLeft columns from the left
            # - if right, add randomRight 0's to the left, remove randomRight columns from the right
            #save the new image to the total dataset (offset digits)

            #pick some random mnist image
            random_image_picker = random.randint(0, 55000)
            tempImage = trainX[random_image_picker]
            tempImage = tempImage.reshape(28,28)

            #random digit betseen 5 and 15
            randomUp = random.randint(5,15)
            randomDown = random.randint(5,15)
            randomLeft = random.randint(5,15)
            randomRight = random.randint(5,15)
            #pick either up, down, left, right, or a combination of up/left, up/right, down/left, down/right
            picker = ["up", "down", "left", "right", "upleft", "upright", "downleft", "downright"]
            picked = np.random.choice(picker)
            if picked == "up":
                for i in range(randomUp):
                    #pad with 0's along the bottom
                    tempImage = np.insert(tempImage, tempImage.shape[0], 0, axis = 0)

                    #delete rows from the top
                    tempImage = np.delete(tempImage, 0, 0)
            if picked == "down":
                for i in range(randomDown):
                    #pad with 0's along the top
                    tempImage = np.insert(tempImage, 0, 0, axis = 0)

                    #delete rows from the bottom
                    tempImage = np.delete(tempImage, tempImage.shape[0] - 1, 0)
            if picked == "left":
                for i in range(randomLeft):
                    #pad with 0's to the right
                    tempImage = np.insert(tempImage, tempImage.shape[1], 0, axis = 1)

                    #delete columns from the left
                    tempImage = np.delete(tempImage, 0, 1)
            if picked == "right":
                for i in range(randomRight):
                    #pad with 0's on the left
                    tempImage = np.insert(tempImage, 0, 0, axis = 1)

                    #delete columns from the right
                    tempImage = np.delete(tempImage, tempImage.shape[1] - 1, 1)
            if picked == "upleft":
                for i in range(randomUp):
                    #pad with 0's along the bottom
                    tempImage = np.insert(tempImage, tempImage.shape[0], 0, axis = 0)

                    #delete rows from the top
                    tempImage = np.delete(tempImage, 0, 0)
                for i in range(randomLeft):
                    #pad with 0's to the right
                    tempImage = np.insert(tempImage, tempImage.shape[1], 0, axis = 1)

                    #delete columns from the left
                    tempImage = np.delete(tempImage, 0, 1)
            if picked == "upright":
                for i in range(randomUp):
                    #pad with 0's along the bottom
                    tempImage = np.insert(tempImage, tempImage.shape[0], 0, axis = 0)

                    #delete rows from the top
                    tempImage = np.delete(tempImage, 0, 0)
                for i in range(randomRight):
                    #pad with 0's on the left
                    tempImage = np.insert(tempImage, 0, 0, axis = 1)

                    #delete columns from the right
                    tempImage = np.delete(tempImage, tempImage.shape[1] - 1, 1)
            if picked == "downleft":
                for i in range(randomDown):
                    #pad with 0's along the top
                    tempImage = np.insert(tempImage, 0, 0, axis = 0)

                    #delete rows from the bottom
                    tempImage = np.delete(tempImage, tempImage.shape[0] - 1, 0)
                for i in range(randomLeft):
                    #pad with 0's to the right
                    tempImage = np.insert(tempImage, tempImage.shape[1], 0, axis = 1)

                    #delete columns from the left
                    tempImage = np.delete(tempImage, 0, 1)
            if picked == "downright":
                for i in range(randomDown):
                    #pad with 0's along the top
                    tempImage = np.insert(tempImage, 0, 0, axis = 0)

                    #delete rows from the bottom
                    tempImage = np.delete(tempImage, tempImage.shape[0] - 1, 0)
                for i in range(randomRight):
                    #pad with 0's on the left
                    tempImage = np.insert(tempImage, 0, 0, axis = 1)

                    #delete columns from the right
                    tempImage = np.delete(tempImage, tempImage.shape[1] - 1, 1)
            tempImage = tempImage.ravel()
            is_it_a_digit = np.concatenate((is_it_a_digit, np.array([tempImage])), axis = 0)
            is_it_a_digit_label = np.concatenate((is_it_a_digit_label, np.array([[0, 1]])), axis = 0) #0 for non-centered digit
        else:
            random_image_picker = random.randint(0, 55000)
            tempImage = trainX[random_image_picker]
            is_it_a_digit = np.concatenate((is_it_a_digit, np.array([tempImage])), axis = 0)
            is_it_a_digit_label = np.concatenate((is_it_a_digit_label, np.array([[1, 0]])), axis = 0) #1 for centered digit

    is_it_a_digit = np.delete(is_it_a_digit, 0, 0)
    is_it_a_digit_label = np.delete(is_it_a_digit_label, 0, 0)
    
    return is_it_a_digit, is_it_a_digit_label

#is_it_a_digit, is_it_a_digit_label = buildOffset(trainX) #Run this line of code to run the above function.

#Save the datasets so they do not have to be rebuilt each time:
import pandas as pd

print(total_input.shape, type(total_input))
total_input_dataframe = pd.DataFrame(total_input)
total_input_dataframe.to_csv("total_input.csv")

print(total_output.shape, type(total_output))
total_output = total_output.reshape(10000, 55)
total_output_dataframe = pd.DataFrame(total_output)
total_output_dataframe.to_csv("total_output.csv")

print(condensed_output.shape, type(condensed_output))
condensed_output_dataframe = pd.DataFrame(condensed_output)
condensed_output_dataframe.to_csv("condensed_output.csv")

print(total_digit_count.shape, type(total_digit_count))
total_digit_count_dataframe = pd.DataFrame(total_digit_count)
total_digit_count_dataframe.to_csv("total_digit_count.csv")

print(total_location.shape, type(total_location))
total_location_dataframe = pd.DataFrame(total_location)
total_location_dataframe.to_csv("total_location.csv")

print(is_it_a_digit.shape, type(is_it_a_digit))
is_it_a_digit_dataframe = pd.DataFrame(is_it_a_digit)
is_it_a_digit_dataframe.to_csv("is_it_a_digit.csv")

print(is_it_a_digit_label.shape, type(is_it_a_digit_label))
is_it_a_digit_label_dataframe = pd.DataFrame(is_it_a_digit_label)
is_it_a_digit_label_dataframe.to_csv("is_it_a_digit_label.csv"
