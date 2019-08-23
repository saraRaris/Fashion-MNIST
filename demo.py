import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import matplotlib.image as mpimg
import skimage
import pdb

from keras.models import model_from_yaml


def init_dict():
    labels_dict = {}
    labels_dict[0] = 'T-shirt/top'
    labels_dict[1] = 'Trouser'
    labels_dict[2] = 'Pullover'
    labels_dict[3] = 'Dress'
    labels_dict[4] = 'Coat'
    labels_dict[5] = 'Sandal'
    labels_dict[6] = 'Shirt'
    labels_dict[7] = 'Sneaker'
    labels_dict[8] = 'Bag'
    labels_dict[9] = 'Ankle boot'
    return labels_dict


def load_model():
    
    #Load model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    
    #Load weights
    model.load_weights('model_weights.h5')
    return model


def resize_im(cropped_im):
    resized = cv2.resize(cropped_im, (28, 28))
    return resized


def inference(image, labels_dict):
    
    #Set correct dimensions for the image
    x = np.expand_dims(image, axis=0)
    x = np.expand_dims(x, axis=-1)
    
    #Give inference
    output = (model.predict(x).tolist())[0]
    
    #Return label if prediction > thr
    if max(output) > 0.90:
        return labels_dict[output.index(max(output))]
    else:
        return None


def merge_candidates(candidates):
    merging = True
    while merging:
        final_cand = []
        merging = False
        for i in range(0, len(candidates)):
            for j in range(i+1, len(candidates)):
                diff = np.mean(abs(np.array([candidates[i]])- np.array([candidates[j]])))
                if diff < 15:
                    merging = True
                    merge_cand = np.column_stack((candidates[i], candidates[j])).tolist()
                    final_cand.append(tuple(list((map(max, merge_cand)))))
                elif candidates[j] not in final_cand:
                    final_cand.append(candidates[j])
            if merging:
                break
            elif candidates[i] not in final_cand:
                final_cand.append(candidates[i])
        candidates = final_cand
    return candidates



def selective_search(frame, gray, model):
    img_lbl, regions = selectivesearch.selective_search(frame, scale=500, sigma=0.9, min_size=10)
    candidates = []

    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.append(r['rect'])

    if candidates:
        #Merges candidates that are very close together
        candidates = merge_candidates(candidates)
    
    return candidates


def capture_video():

    cap = cv2.VideoCapture(0)

    # Get the Default resolutions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and filename.
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    count_frames = 0
    
    while(count_frames < 40):
        ret, frame = cap.read()
        count_frames += 1
        if ret==True:
            # write frame
            out.write(frame)
            
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    
    #Dictionary with Fashion-MNIST dataset labels
    labels_dict = init_dict()
    
    #Records 10 frame video
    capture_video()
    
    #Loads model
    model = load_model()
    
    #Opens recorded video
    cap = cv2.VideoCapture('output.avi')
    
    #Initialize final processed video
    processed = cv2.VideoWriter('processed_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cap.get(3)),int(cap.get(4))))
    
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is not None:
            #Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #Perform selective search
            candidates = selective_search(frame, gray, model)

            #Initialize detection list
            detections = []
            labels = []
            
            #Crop image
            for x, y, w, h in candidates:
                cropped_image = gray[y:y+h , x:x+w]
                image = resize_im(cropped_image)
                
                #Give inference
                label = inference(image, labels_dict)
                if label:
                    labels.append(label)
                    detections.append([x, y, w, h])
        
            #Draw rectangles on the original frame
            for i in range(0, len(detections)):
                x, y, w, h = detections[i]
                outlined_image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 5)
                frame = cv2.putText(outlined_image, labels[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2, cv2.LINE_AA)

            cv2.imshow('image',frame)
            cv2.waitKey(1)
            
            #Saves processed frame in video
            processed.write(frame)

            #Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            processed.release()
            cap.release()
            cv2.destroyAllWindows()
            break



