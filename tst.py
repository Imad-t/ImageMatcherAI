import cv2
import os
import numpy as np


def extract_sift_descriptor(image_path, target_size=(100, 100)):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)

    return descriptors

def calculate_sift_descriptors_in_directory(directory, target_size=(100, 100)):
    descriptors_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(directory, filename)
            descriptors = extract_sift_descriptor(image_path, target_size)
            descriptors_dict[filename] = descriptors

    return descriptors_dict


def find_most_similar_image(query_image_path, training_descriptors_dict):
    query_descriptors = extract_sift_descriptor(query_image_path)

    similarities = {}

    for filename, training_descriptors in training_descriptors_dict.items():
        #bruteforce matcher object
        bf = cv2.BFMatcher()
        #k-nearest neighbors matching
        matches = bf.knnMatch(query_descriptors, training_descriptors, k=2)

        #ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        #calculate the similarity percentage
        similarity_percentage = (len(good_matches) / len(query_descriptors)) * 100
        similarities[filename] = similarity_percentage

    # Find the filename with the highest resemblance
    most_similar_image = max(similarities, key=similarities.get)
    return most_similar_image, similarities[most_similar_image]

def draw_key_points(img):
    resized_img = cv2.resize(img, (640,400))
    gray= cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(resized_img,None)
    img = cv2.drawKeypoints(gray,kp,resized_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def main():
    
    
    # Choose a query image from the test set
    query_image_path = 'dataset/test/vegetables/spo_010.jpg'

    if 'fruits' in query_image_path:
        training_path = 'dataset/training/Fruits'
    elif 'vegetables' in query_image_path:
        training_path = 'dataset/training/Vegetables'
    else:
        print("Invalid category in the query image pathname.")
        return
    
    # Calculate SIFT descriptors for training images
    training_descriptors_dict = calculate_sift_descriptors_in_directory(training_path)

    # Find the most similar image in the training set
    most_similar_image, resemblance_percentage = find_most_similar_image(query_image_path, training_descriptors_dict)

    # Display the query and most similar images side by side
    query_image = cv2.imread(query_image_path)
    similar_image_path = os.path.join(training_path, most_similar_image)
    similar_image = cv2.imread(similar_image_path)

    cv2.imshow('Query Image', draw_key_points(query_image))
    cv2.imshow('Most Similar Image', draw_key_points(similar_image))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
