import os
import torch
from PIL import Image
import torchvision
import torchvision.transforms as T
import cv2


def main():
    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_detector = torch.load(
        os.path.join('saved_models', 'person_detector.pt'),
        map_location=device)

    person_detector.to(device)
    person_detector.eval()

    clothes_detector = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=os.path.join('saved_models', 'clothes_detector.pt'))

    clothes_detector.conf = 0.6  # confidence threshold (0-1)
    clothes_detector.iou = 0.4  # NMS IoU threshold (0-1)

    # input image
    img_path = os.path.join('test_images', 'test.jpg')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    people = detect_people(img, person_detector, device)  # people is now an array of cv2 images, each one containing 1 detected person

    # NOTE: running clothes detector only if a person is detected
    if len(people) > 0:
        clothing = detect_clothing_yolo(people, clothes_detector)  # yolov5 object detection
        # clothing is now an array of cv2 images, each one showing the clothing items for each person

        for i, image in enumerate(clothing):
            img = Image.fromarray(image.astype('uint8')).convert('RGB')
            img.save(os.path.join('output', f'output_{i}.jpg'))
            print(f'Saved output_{i}.jpg')


def detect_people(image, person_detector, device):
    image_copy = image.copy()  # used later while cropping

    preprocess = T.Compose([
        T.ToTensor()
    ])

    image = preprocess(image)

    with torch.no_grad():
        prediction = person_detector([image.to(device)])

    boxes = prediction[0]["boxes"]
    scores = prediction[0]["scores"]
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    boxes = boxes.tolist()
    scores = scores.tolist()

    threshold = 0.6
    final_boxes = []
    final_scores = []

    people = []
    for i in keep:
        if scores[i] < threshold:
            continue
        else:
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            x1, y1, x2, y2 = map(int, boxes[i])
            cropped = image_copy[y1:y2, x1:x2]
            people.append(cropped)

    # print(final_boxes)
    # print(final_scores)

    return people


def detect_clothing_yolo(people, clothes_detector):
    imgs = []
    for person in people:
        # decided to add padding because the labels drawn by yolo usually don't fit into the image
        padded = cv2.copyMakeBorder(person, top=0, bottom=0, left=0, right=250,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[16, 21, 24])
        imgs.append(Image.fromarray(padded.astype('uint8')))

    results = clothes_detector(imgs, size=640)
    results.render()

    return results.imgs


if __name__ == "__main__":
    main()
