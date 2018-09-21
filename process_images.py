import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os

steering_correction = .2

def cropImage(img):
    return img[35:140, 0:320]


def resizeImage(img, dst_x, dst_y):
    return cv2.resize(img, (dst_x, dst_y))


def gausNoise(image, prob):
    noise = np.zeros(image.shape, np.uint8)
    m = (0, 0, 0)
    s = (255 * prob, 255 * prob, 255 * prob)
    cv2.randn(noise, mean=m, stddev=s)
    return noise + image


def equalizeImage(img):
    for i in range(3):
        img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    return img


def gausBlur(img):
    kernel = np.ones((3, 3), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)


def normalizeImage(img):
    return cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)


def showImage(img, title=""):
    plt.figure(figsize=(20,20))
    plt.title(title)
    plt.imshow(img)
    #plt.savefig("out_images/sample_"+title+".jpg")
    plt.show()
    cv2.imwrite("out_images/sample_"+title+".jpg", img)


def getImage(path, show=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return process_image(img, show)


def process_image(img, show=False):
    if show:
        showImage(img, "Original")
    img = cropImage(img)
    if show:
        showImage(img, "Cropped")
    img = resizeImage(img, 200, 66)
    if show:
        showImage(img, "Resized")
    img = gausNoise(img, 0.1)
    if show:
        showImage(img, "GaussianNoised")
    img = equalizeImage(img)
    if show:
        showImage(img, "HistogramEqualized")
    img = gausBlur(img)
    if show:
        showImage(img, "GaussianBlurred")
    img = normalizeImage(img)
    if show:
        showImage(img, "Normalized")
    return img


def flip(img):
    return cv2.flip(img, 1)

def read_logs(dir):
    '''
    Read
    :param dir:
    :return:
    '''
    lines = []
    csv_file = dir + "/" + 'driving_log.csv'
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def checkSteeringAngle(angle):
    return True
    if abs(angle) < 0.1:
        if random.randint(1, 2) == 1:
            return True
        else:
            return False
    elif abs(angle) > 0.7:
        return False
    else:
        return True



def process_all_images( dir ):

    lines = read_logs(dir)

    angles = []

    filenames = []

    save_dir_path = 'data/result/'

    for line in lines:

        new_images = []

        # augment images and angles

        center_image_path = line[0].replace(" ", "")
        left_image_path = line[1].replace(" ", "")
        right_image_path = line[2].replace(" ", "")

        center_image_name = os.path.basename(center_image_path)
        left_image_name = os.path.basename(left_image_path)
        right_image_name = os.path.basename(right_image_path)

        base_filenames = [center_image_name,left_image_name,right_image_name]

        center_angle = float( line[3] )
        if checkSteeringAngle(center_angle):

            # adjust steers of left and right cameras
            left_angle = center_angle + steering_correction
            #left_angle = min(left_angle,1)
            right_angle = center_angle - steering_correction
            #right_angle = max ( right_angle, -1 )

            #print("Try to get image {}".format(center_image_path))
            center_img = getImage(center_image_path)
            left_img = getImage(left_image_path)
            right_img = getImage(right_image_path)

            flip_center = flip(center_img)
            flip_left = flip(left_img)
            flip_right = flip(right_img)

            base_filenames += ['flip_' + fname for fname in base_filenames]

            save_filenames = [ save_dir_path + fname for fname in base_filenames ]

            new_images.extend([center_img,left_img,right_img])
            new_images.extend([flip_center,flip_left,flip_right])

            angles.extend([center_angle,left_angle,right_angle])
            angles.extend([-center_angle,-left_angle,-right_angle])

            filenames += save_filenames

            for i in range(len(new_images)):
                cv2.imwrite(save_filenames[i], new_images[i])

    return angles,filenames

def saveResult( angles, filenames  ):
    with open('data/result_data.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(filenames)):
            writer.writerow([filenames[i]] + [angles[i]])

if __name__ == '__main__':
    hist_angles = []
    filenames = []

    hist_angles1 = []
    hist_angles2 = []
    hist_angles3 = []
    hist_angles4 = []

    filenames1 = []
    filenames2 = []
    filenames3 = []
    filenames4 = []

    hist_angles1,filenames1 = process_all_images('races/test2/')
    hist_angles2,filenames2 = process_all_images('races/test3/')
    hist_angles3,filenames3 = process_all_images('races/test5/')
    hist_angles4,filenames4 = process_all_images('races/test6/')

    hist_angles = hist_angles1 + hist_angles2 + hist_angles3 + hist_angles4
    filenames = filenames1 + filenames2 + filenames3 + filenames4


    print("Number of images: {}".format(len(filenames)))
    print("Number of steering angles: {}".format(len(hist_angles)))

    num_bins = 27 # <- number of bins for the histogram


    avg_angles_per_bin = len(hist_angles) / num_bins



    print("Average angles per bins: {}".format(avg_angles_per_bin))
    hist, bins = np.histogram(hist_angles, num_bins)

    center = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(20,20))
    plt.hist(hist_angles,num_bins)
    plt.plot((np.min(hist_angles), np.max(hist_angles)), (avg_angles_per_bin, avg_angles_per_bin), 'k-')
    plt.savefig("out_images/angles_hist.jpg")
    #plt.show()

    keep_prob = []

    for i in range(num_bins):
        if ( hist[i] < avg_angles_per_bin*0.4 ):
            keep_prob.append(1.0)
        else:
            prob = 1.0/(hist[i]/(avg_angles_per_bin*0.4))
            keep_prob.append(prob)



    to_delete = []
    for i in range(len(hist_angles)):
        for j in range(num_bins):
            if hist_angles[i] > bins[j] and hist_angles[i] <= bins[j+1]:
                if np.random.rand() > keep_prob[j]:
                    to_delete.append(i)

    print('Numbers to delete: {} -> {}%'.format(len(to_delete),100*len(to_delete)/len(hist_angles)))

    filenames = np.delete(filenames, to_delete, axis=0)
    hist_angles = np.delete(hist_angles, to_delete)

    hist, bins = np.histogram(hist_angles, num_bins)

    plt.figure(figsize=(20,20))
    center = (bins[:-1] + bins[1:]) / 2
    plt.hist(hist_angles,num_bins)
    plt.plot((np.min(hist_angles), np.max(hist_angles)), (avg_angles_per_bin, avg_angles_per_bin), 'k-')
    plt.savefig("out_images/angles_hist_normal.jpg")
    #plt.show()

    print("Save result")

    saveResult(hist_angles,filenames)

    print("Number of images: {}".format(len(filenames)))
    print("Number of steering angles: {}".format(len(hist_angles)))




















