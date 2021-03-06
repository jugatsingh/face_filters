import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import imutils


def get_image(data_dir, file_name):
    path = os.path.join(data_dir, file_name)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def convert_image_to_bw(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return image


def detect_keypoints(model, image, faces, padding=50):
    img_height, img_width = image.shape[0], image.shape[1]
    images, keypoints = [], []
    for coords in faces:
        img = image[max(0, coords[1]-padding):
                    min(coords[1]+coords[-1]+padding, img_height),
                    max(0, coords[0]-padding):
                    min(coords[0]+coords[2]+padding, img_width)]
        img = (img/255.0).astype(np.float32)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.rollaxis(img, 2, 0)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        results = model.forward(img)
        results = results.view(results.size()[0], 68, -1).cpu()
        pred = results[0].cpu().data
        pred = pred.numpy()
        pred = pred * 50 + 100
        keypoints.append(pred)
    return images, keypoints


def detect_faces(classifier, image):
    rects = classifier.detectMultiScale(image, 1.2, 2)
    for x, y, w, h in rects:
        cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    return rects


def visualize_output(faces, test_outputs):
    for i, face in enumerate(faces):
        plt.figure(figsize=(5, 5))
        plt.imshow(face)
        plt.scatter(test_outputs[i][:, 0], test_outputs[i][:, 1], s=20, marker='.', c='m')
        plt.axis('off')
    plt.show()


def get_part_filter(img, model, scale=1.1, neighbors=5, width=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = model.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(width, width),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return features


def get_classifier(data_dir, file_name):
    path = os.path.join(data_dir, file_name)
    classifier = cv2.CascadeClassifier(path)
    return classifier


def apply_dog_filter(face, filt):
    filt_h, filt_w, _ = filt.shape
    face_h, face_w, _ = face.shape
    factor = min(face_h/filt_h, face_w/filt_w)
    new_filt_h, new_filt_w = int(filt_h*factor), int(filt_w*factor)
    new_filt_shape = (new_filt_w, new_filt_h)
    resized_filter = cv2.resize(filt, new_filt_shape)
    masked_face = face.copy()
    non_white_pixels = (resized_filter < 250).all(axis=2)
    offset_h, offset_w = int((face_h-new_filt_h)/2), int((face_w-new_filt_w)/2)
    masked_face[offset_h:offset_h+new_filt_h, offset_w:offset_w +
                new_filt_w][non_white_pixels] = \
        resized_filter[non_white_pixels]
    return masked_face


def dog_face_filter(image, classifier, dog_filter):
    image_h, image_w = image.shape[0], image.shape[1]
    rectangles = detect_faces(classifier=classifier, image=image)
    for x, y, w, h in rectangles:
        y0, y1 = int(y - 0.25*h), int(y + 0.75*h)
        x0, x1 = x, x + w
        if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
            continue
        image[y0: y1, x0: x1] = apply_dog_filter(image[y0: y1, x0: x1], dog_filter)
    return image


def overlay_filter_on_image(img, filt, off_x, off_y):
    (h, w) = (filt.shape[0], filt.shape[1])
    (imgH, imgW) = (img.shape[0], img.shape[1])
    if off_y+h >= imgH:
        filt = filt[0:imgH-off_y, :, :]
    if off_x+w >= imgW:
        filt = filt[:, 0:imgW-off_x, :]
    if off_x < 0:
        filt = filt[:, abs(off_x)::, :]
        w = filt.shape[1]
        off_x = 0
    for c in range(3):
        img[off_y:int(off_y+h), off_x:int(off_x+w), c] =  \
            filt[:, :, c] * (filt[:, :, 3]/255.0) + \
            img[off_y:off_y+h, off_x:off_x+w, c] * (1.0 - filt[:, :, 3]/255.0)
    return img


def apply_sunglasses_filter(img, filt, part_filter, off_x, off_y, off_y_image,
                            actual_width, x, y, w, h):
    (filt_h, filt_w) = (filt.shape[0], filt.shape[1])
    xpos = x + off_x
    ypos = y + off_y
    factor = 1.0 * actual_width/filt_w
    sub_img = img[y+off_y_image:y+h, x:x+w, :]
    feature = get_part_filter(sub_img, part_filter, 1.3, 10, 10)
    if len(feature) != 0:
        xpos, ypos = x, y + feature[0, 1]
    filt = cv2.resize(filt, (0, 0), fx=factor, fy=factor)
    img = overlay_filter_on_image(img, filt, xpos, int(ypos))
    return img


def apply_mustache_filter(img, filt, part_filter, off_x, off_y, off_y_image,
                          actual_width, x, y, w, h):
    (filt_h, filt_w) = (filt.shape[0], filt.shape[1])
    xpos = x + off_x
    ypos = y + off_y
    factor = 1.0 * actual_width/filt_w
    sub_img = img[y+off_y_image:y+h, x:x+w, :]
    feature = get_part_filter(sub_img, part_filter, 1.3, 10, 10)
    if len(feature) != 0:
        xpos, ypos = x, y + feature[0, 1]
        size_mustache = 1.2
        factor = 1.0*(feature[0, 2]*size_mustache)/filt_w
        xpos = x + feature[0, 0] - int(feature[0, 2]*(size_mustache-1)/2)
        ypos = y + off_y_image + feature[0, 1] - int(filt_h*factor)
    filt = cv2.resize(filt, (0, 0), fx=factor, fy=factor)
    img = overlay_filter_on_image(img, filt, xpos, int(ypos))
    return img


def apply_flag_filter(frame, flag, keypts, face_loc):
    keypts = keypts[0]
    flag_height = int(abs(keypts[28][1] - keypts[30][1]))
    flag_width = int(abs(keypts[48][0] - keypts[54][0])/2)
    h, w, _ = flag.shape
    r = flag_height / float(h)
    dim = (int(w * r), flag_height)
    scaled_flag = cv2.resize(flag, dim, interpolation=cv2.INTER_AREA)
    eyebrow_height_diff = abs(keypts[17][1] - keypts[26][1])
    eyebrow_width = abs(keypts[17][0] - keypts[26][0])
    eyebrow_angle = np.arctan(eyebrow_height_diff/eyebrow_width)*(180/np.pi)
    eyebrow_angle = eyebrow_angle if keypts[17][1] < keypts[26][1] else -eyebrow_angle
    new_flag = imutils.rotate_bound(scaled_flag, eyebrow_angle)
    b = (np.copy(new_flag))[:, :, :3]/255
    flag_x = int(keypts[35][0] + abs(keypts[33][0] - keypts[35][0])*5/2)
    flag_y = int((keypts[33][0] + keypts[30][0])/2)
    a = images[0][flag_y: flag_y + dim[1], flag_x: min(flag_x + dim[0], frame.shape[1])]
    a_copy = np.copy(a)
    a = a/255
    a = np.where(a < 0.5, 2*a*b, 1 - 2*(1-a)*(1-b)) * 255
    a = np.where(np.expand_dims(new_flag[:, :, 3], axis=2) < 5, a_copy, a)
    images[0][flag_y: flag_y + dim[1], flag_x: min(flag_x + dim[0], frame.shape[1])] = a
    frame[face_loc[1]:face_loc[1] + face_loc[3], face_loc[0]: face_loc[0] + face_loc[2]
          ] = cv2.resize(images[0], (face_loc[2], face_loc[3]), interpolation=cv2.INTER_AREA)
    return frame


def get_eyes_filter():
    return cv2.CascadeClassifier('./models/eye.xml')


def get_emotion_predictor(model_path):
    net = emotion_model.Net().float().to(device)
    pretrained_model = torch.load(model_path, map_location='cpu')
    net.load_state_dict(pretrained_model)
    net.eval()
    return net


def emotion_predictor(emotion_classifier, image):
    image = convert_image_to_bw(image)
#     image = cv2.resize(image, (48, 48)).reshape((1, 1, 48, 48))
    image = cv2.resize(image, (48, 48)).reshape((1, 48, 48, 1)).astype("float")/255.0
#     X = torch.from_numpy(image).float().to(device)
    preds = emotion_classifier.predict(image)[0]
    emotion_probability = np.max(preds)
    return preds.argmax()
    return np.argmax(net(X).data.cpu().numpy(), axis=1)[0]
