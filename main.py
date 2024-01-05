import cv2
import os

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
        # brute-force matcher object
        bf = cv2.BFMatcher()
        # k-nearest neighbors matching
        matches = bf.knnMatch(query_descriptors, training_descriptors, k=2)
        # ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # calculate the similarity percentage
        similarity_percentage = (len(good_matches) / len(query_descriptors)) * 100
        similarities[filename] = similarity_percentage

    # find the most similar image
    most_similar_image = max(similarities, key=similarities.get)
    return most_similar_image


def draw_matches(query_image, similar_image):
    max_width = 640
    max_height = 400

    if (query_image.shape[1] > max_width or query_image.shape[0] > max_height) or (
            similar_image.shape[1] > max_width or similar_image.shape[0] > max_height):
        query_image = cv2.resize(query_image, (max_width, max_height))
        similar_image = cv2.resize(similar_image, (max_width, max_height))

    sift = cv2.SIFT_create()
    query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
    similar_keypoints, similar_descriptors = sift.detectAndCompute(similar_image, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(query_descriptors, similar_descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    img_matches = cv2.drawMatches(query_image, query_keypoints, similar_image, similar_keypoints, good_matches, None)

    return img_matches


def main():
    query_image_path = 'dataset/test/fruits/wat_004.jpg'

    if 'fruits' in query_image_path:
        training_path = 'dataset/training/Fruits'
    elif 'vegetables' in query_image_path:
        training_path = 'dataset/training/Vegetables'
    else:
        print("Invalid category in the query image pathname.")
        return

    training_descriptors_dict = calculate_sift_descriptors_in_directory(training_path)

    most_similar_image = find_most_similar_image(query_image_path, training_descriptors_dict)

    query_image = cv2.imread(query_image_path)
    similar_image_path = os.path.join(training_path, most_similar_image)
    similar_image = cv2.imread(similar_image_path)

    img_matches = draw_matches(query_image, similar_image)

    cv2.imshow(f'Most similar image : {most_similar_image}', img_matches)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
