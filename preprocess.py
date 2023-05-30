import cv2
import os

digits = [0,1,2,3,4,5,6,7,8,9]

for digit in digits:
    os.makedirs('dataset_preprocessed_train_test\\'+ str(digit) +'\\', exist_ok=True)
#
if __name__ == "__main__":
    for digit in digits:
        print('preprocessing ', digit)
        for filename in os.listdir('dataset_train\\'+ str(digit) +'\\'):
            img = cv2.imread('dataset_train\\'+ str(digit) +'\\'+filename)
            # print(img.shape)
            img = 255 - img
            # cv2.imshow('ImageWindow', img)
            # cv2.waitKey()
            # exit()
            # print(img)
            gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(gray_scale_img.shape)
            # exit()
            ret, mask = cv2.threshold(gray_scale_img, 180, 255, cv2.THRESH_BINARY)
            final_img = cv2.bitwise_and(gray_scale_img, gray_scale_img, mask=mask)
            ret, new_img = cv2.threshold(final_img, 180, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,
                                                                 1))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
            dilated_img = cv2.dilate(new_img, kernel, iterations=1)  # dilate , more the iteration more the dilation
            # cv2.imshow('ImageWindow', dilated_img)
            # cv2.waitKey()
            # exit()
            contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_img = []
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                contours_img.append([x, y, w, h])

            contours_img.sort()
            i = 0
            digit_segment_img = []
            for contour in contours_img:
                # get rectangle bounding contour
                [x, y, w, h] = contour
                # eliminating false positive from our contour
                if w < 20 and h < 20:
                    continue
                # drawing rectangle around contour
                rec_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # cv2.imshow("ImageWindow", rec_img)
                # cv2.waitKey()
                # exit()
                # crop each contour and save individually
                cropped_img = final_img[y:y + h, x:x + w]
            final_img = cv2.resize(cropped_img, (32, 32))
            cv2.imwrite('dataset_preprocessed_train_test\\'+str(digit)+'\\'+filename, final_img)
# #