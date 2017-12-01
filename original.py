# TrainAndTest.py

import cv2
import numpy as np
import operator
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():


    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True

###################################################################################################
def main():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    cap = cv2.VideoCapture(0)

    klatki = 0
    while (cap.isOpened()):
        ret, imgTestingNumbers = cap.read()

        if klatki == 25:
            #imgTestingNumbers = cv2.imread("test3.png")

            if imgTestingNumbers is None:
                print ("error: image not read from file \n\n"  )
                os.system("pause")
                return
            # end if

            imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
            imgBlurred = cv2.GaussianBlur(imgGray, (15,15), 0)

            imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                              255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV,
                                              21,
                                              3)
            #imgCanny = cv2.Canny(imgThresh, 0, 50)
            cv2.imshow("imgThresh", imgThresh)
            imgThreshCopy = imgThresh.copy()

            imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)
            #cv2.imshow("imgContours", imgContours)
            for npaContour in npaContours:
                contourWithData = ContourWithData()
                contourWithData.npaContour = npaContour
                contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
                contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
                contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
                allContoursWithData.append(contourWithData)
            # end for

            for contourWithData in allContoursWithData:
                if contourWithData.checkIfContourIsValid():
                    validContoursWithData.append(contourWithData)
                # end if
            # end for

            validContoursWithData.sort(key = operator.attrgetter("intRectX"))

            strFinalString = ""

            for contourWithData in validContoursWithData:
                                                        # draw a green rect around the current char
                cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                              (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                              (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                              (0, 255, 0),              # green
                              2)                        # thickness

                imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                                   contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

                npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

                npaROIResized = np.float32(npaROIResized)

                retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)

                strCurrentChar = str(chr(int(npaResults[0][0])))

                strFinalString = strFinalString + strCurrentChar
            #cv2.imshow("imgTesting",
                      # imgTestingNumbers)  # show input image with green boxes drawn around found digits
            # end for

            print ("\n" + strFinalString + "\n")

            klatki = 0
            allContoursWithData = []
            validContoursWithData = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            #  wait for user key press
        klatki = klatki + 1
        cv2.imshow("imgTestingNumbers",
                      imgTestingNumbers)  # show input image with green boxes drawn around found digits
        imgTestingNumbers = []

    cap.release()
    cv2.destroyAllWindows()  # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if









