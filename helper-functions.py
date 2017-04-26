def absoluteAnswer(predictionArray):
    """
    input is an array of values
    function finds the largest value, assigns it to 1, 0's elsewhere
    output is an array of 0's and 1
    """
    final_out = []
    max_value = 0
    for i in predictionArray:
        if i >= max_value:
            max_value = i
    for i in predictionArray:
        if i == max_value:
            final_out.append(1.)
        else:
            final_out.append(0.)
    return final_out

def absoluteAnswerLarge(predictionArray):
    """
    input is an array
    iterates through each row and one-hot encodes it
    output is the full combined array
    """
    for i in range(len(predictionArray)):
        predictionArray[i] = absoluteAnswer(predictionArray[i])
    
    return predictionArray
    
def plotImage(array):
    """
    input is an array that is shaped as the image dimensions
    output is the plotted image
    """
    plt.imshow(array)
    plt.show()
    
def setData(random_indices, x_data):
    """
    input is the dataset of MNIST images
    output[0] is the newly stitched data
    output[1] is the (x,y) coordinates of each digit in each image as a np array
    """
    #pull number_of_digits numbers from the dataset with indices random_indices
    sequenceInput = np.array([[]]*80)
    originalDimensions = np.array([[0, 0, 28, 28], [0, 29, 28, 56], [0, 57, 28, 84], [0, 85, 28, 112], [0, 113, 28, 140]])
    #originalDimensions = np.array([[14, 14], [14, 42], [14, 70], [14, 98], [14, 126]]) #just the centers
    leftSpacer = []
    upDownSpacer = []
    
    for i in range(len(random_indices)):
        #resize each digit
        newDigit = x_data[random_indices[i]].reshape([28, 28])
        
        #add black space above and below each digit
        #random digit between 1 and 52
        randomSpacer = random.randint(1,52)
        upDownSpacer.append(randomSpacer)
        
        #add randomSpacer black lines above
        for j in range(randomSpacer):
            newDigit = np.append(np.array([np.zeros((28))]), newDigit, axis = 0)
        
        #add 12-randomSpacer black lines below
        for k in range(52 - randomSpacer):
            newDigit = np.append(newDigit, np.array([np.zeros((28))]), axis = 0)
        
        #add black space to the left of each digit
        randomSpacerLeft = random.randint(1,20)
        leftSpacer.append(randomSpacerLeft)
        
        for l in range(randomSpacerLeft):
            newDigit = np.insert(newDigit, 0, 0, axis = 1)
        
        #add the new array to total array
        sequenceInput = np.concatenate((sequenceInput, newDigit), axis = 1)
        
    #to do the spacing numbers, make sure you stack the adds. As in, the first number gets shifted by the first randomspacerleft
    #the second numbers gets shifted by the first randomspacerleft AND the second randomspacerleft. etc.
        originalDimensions[i][0] += randomSpacer
        originalDimensions[i][2] += randomSpacer #only used when we use corners
        originalDimensions[i][1] += sum(leftSpacer[0:i+1])
        originalDimensions[i][3] += sum(leftSpacer[0:i+1]) #only used when we use corners
    
    return sequenceInput, originalDimensions
    
def saveOutput(number_of_digits, y_data, random_indices, new_output):
    """
    input is the number of digits, the label data, the indices of the mnist images that were stitched together, and the output array
    output[0] is the output array, and output[1] is the condensed output array
    """
    sequenceOutput = np.zeros(11)
    newSequenceOutput = [0,0,0,0,0,0,0,0,0,0]
    for i in range(number_of_digits):
        savedSolution = y_data[random_indices[i]]
        predictionArray = np.array(savedSolution)            
        predictionArrayWithZero = np.array(np.concatenate((predictionArray, np.array([0])), axis = 0))
        predictionArrayWithZero = np.array(absoluteAnswer(predictionArrayWithZero))
        sequenceOutput = np.concatenate((sequenceOutput, predictionArrayWithZero), axis = 0)
                
        #save the new_output
        for j in range(len(savedSolution)):
            if absoluteAnswer(savedSolution)[j] == 1:
                newSequenceOutput[j] = 1
    
    new_output = np.concatenate((new_output, np.array([newSequenceOutput])), axis = 0)
    
    sequenceOutput = sequenceOutput.reshape([number_of_digits + 1, 11])
    sequenceOutput = np.delete(sequenceOutput, 0, 0)
    sequenceOutput = sequenceOutput.ravel()
    
    while len(sequenceOutput) < 55:
        sequenceOutput = np.concatenate((sequenceOutput, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])), axis = 0)
        
    return sequenceOutput, new_output
    
def runBuild():
    """
    build all of the data: the input images, the output arrays, condensed output arrays, the number of digits, and the location of the digits
    """
    totalInput = np.array([range(TOTAL_IMAGE_DIMENSIONS[0] * TOTAL_IMAGE_DIMENSIONS[1])])
    totalOutput = np.array([range(55)])
    newOutput = np.array([range(10)])
    totalDigitCount = np.array([range(5)])
    totalLocation = np.array([range(20)])
    
    iterationCounter = 0
    for iteration in range(10000): #we will make 10,000 images to try and classify
        iterationCounter += 1
        #randomly select 1, 2, 3, 4, or 5 (the number of digits to use)
        number_of_digits = np.random.choice([1, 2, 3, 4, 5])
        
        #save the number of digits:
        digitCounter = [0,0,0,0,0]
        digitCounter[number_of_digits - 1] = 1
        totalDigitCount = np.concatenate((totalDigitCount, np.array([np.array(digitCounter)])), axis = 0)
        
        #randomly select number_of_digits amount of indices from the training dataset size (55000)
        random_indices = random.sample(range(0, 55000), number_of_digits)
        sequenceInput, newLocations = setData(random_indices, trainX)
        
        #save the locations of the digits
        newLocations = newLocations.ravel()
        totalLocation = np.concatenate((totalLocation, np.array([newLocations])), axis = 0)
    
        #save the output
        sequenceOutput, newOutput = saveOutput(number_of_digits, trainY, random_indices, newOutput)
        
        #fill in the rest of the image with black space to preserve the size of the inputs    
        zeroBuffer = np.zeros((TOTAL_IMAGE_DIMENSIONS[0], TOTAL_IMAGE_DIMENSIONS[1] - sequenceInput.shape[1]))
        resized = np.concatenate((sequenceInput, zeroBuffer), axis = 1) 
    
        #flatten the image to a single array
        resized = resized.ravel()
    
        #add the input to the total input matrix
        totalInput = np.concatenate((totalInput, np.array([resized])), axis = 0)
        
        #add the output to the total output matrix
        totalOutput = np.concatenate((totalOutput, np.array([sequenceOutput])), axis = 0)
        
        
        if iterationCounter%100 == 0:
            print(str(iterationCounter / 100), "percent complete. ", sep = ' ', end='', flush=True)
    
    totalInput = np.delete(totalInput, 0, 0)
    
    totalOutput = totalOutput.reshape([10001, 5, 11])
    totalOutput = np.delete(totalOutput, 0, 0)
    print("output size: ", totalOutput.shape)
    
    #newoutput
    newOutput = np.delete(newOutput, 0, 0)
    print("new output size: ", newOutput.shape)
    
    #digit counter
    totalDigitCount = np.delete(totalDigitCount, 0, 0)
    
    #digit locations
    totalLocation = np.delete(totalLocation, 0, 0)
    print("total location output size: ", totalLocation.shape)
    return totalInput, totalOutput, newOutput, totalDigitCount, totalLocation
    
def numberOfDigits(array):
    """
    input is a one-hot encoded array, and the output is the integer representing the number of digits
    for instance, [0,1,0,0,0] outputs 2
    """
    total_digit_count = 0
    for arrayLength in range(len(array)):
        if array[arrayLength] == 1:
            total_digit_count = arrayLength + 1
    return total_digit_count

#y_data = y_dtrain
#n_digit_info = y_ctrain
def plotLocations(y_data, n_digit_info):
    """
    from the location info in y_data, build an array that can be plotted with the location info as dots
    """
    for i in range(len(y_data)):
    #for i in range(10):
        tempArray = y_data[i]
        n_digits = numberOfDigits(n_digit_info[i])
        deleteStart = 4*n_digits
        deleteEnd = len(y_data[i])
        tempArray = np.delete(tempArray, range(deleteStart, deleteEnd), axis = 0)
    
        #plot the info from tempArray onto an 80x240 black image
        locationGraphOutput = np.zeros(19200)
        locationGraphOutput = locationGraphOutput.reshape(TOTAL_IMAGE_DIMENSIONS)
        for k in range(len(tempArray)):
            if k%2 == 0:
                kMinus1 = y_data[i][k] - 1
                kNextMinus1 = y_data[i][k+1] - 1
                locationGraphOutput[kMinus1][kNextMinus1] = 225
        
        locationGraphOutput = locationGraphOutput.ravel()
        totalLocationGraph = np.concatenate((totalLocationGraph, np.array([locationGraphOutput])), axis = 0)
    totalLocationGraph = np.delete(totalLocationGraph, 0, 0)
    return totalLocationGraph
    
def removeExtraLocation(y_data, n_digit_info):
    """
    from the location info, change extra coordinates to -1 to specify if a digit does not exist
    """
    new_y_data = y_data #so I don't have to rebuild the entire dataset if I mess up

    for i in range(len(new_y_data)):
        n_values = numberOfDigits(n_digit_info[i])
        
        deleteStart = 4*n_values
        
        deleteEnd = len(y_data[i])
        
        new_y_data[i][deleteStart:deleteEnd] = -1
        
    return new_y_data
