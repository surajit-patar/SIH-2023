import numpy as np
import cv2 as cv
import tensorflow as tf
from motrackers.detectors import YOLOv3
from motrackers import  IOUTracker
from motrackers.utils import draw_tracks
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import time
from collections import defaultdict
import os
import face_recognition
import datetime


CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.2
DRAW_BOUNDING_BOXES = True
USE_GPU = False

VIDEO_FILE = "./VIDEOS/col_sort.mp4"
WEIGHTS_PATH = './MODELS/pretrained_models/yolo_weights/yolov3.weights'
CONFIG_FILE_PATH = './MODELS/pretrained_models/yolo_weights/yolov3.cfg'
LABELS_PATH = "./MODELS/pretrained_models/yolo_weights/coco_names.json"

temporalClusterDirection = [] 

model = YOLOv3(
    weights_path=WEIGHTS_PATH,
    configfile_path=CONFIG_FILE_PATH,
    labels_path=LABELS_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    draw_bboxes=DRAW_BOUNDING_BOXES,
    use_gpu=USE_GPU
)


tracker = IOUTracker(max_lost=30, iou_threshold=0.02, min_detection_confidence=0.4, max_detection_confidence=0.7,tracker_output_format='mot_challenge')


past_location = {0:[0,0]}


def get_face_encodings():
    

    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []


    persons_dir = "PERSONS"


    for folder_name in os.listdir(persons_dir):
        folder_path = os.path.join(persons_dir, folder_name)
        
# Skip any files that might be in the 'PERSONS' directory
        if not os.path.isdir(folder_path):
            continue
        
# Loop over the images in each 'OFFENDER' directory
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
# Learn how to recognize the image by encoding it
            current_image = face_recognition.load_image_file(image_path)
            current_face_encoding = face_recognition.face_encodings(current_image)
            
# Skip images that couldn't be encoded
            if not current_face_encoding:
                print(f"No faces found in the image {image_path}")
                continue
            
# Add the face encoding to the array of known faces
            known_face_encodings.append(current_face_encoding[0])
            known_face_names.append(folder_name)
            
    return known_face_encodings, known_face_names


def get_direction_vector(past_location, current_location):
    x_diff = current_location[0] - past_location[0]
    y_diff = current_location[1] - past_location[1]
    

    if y_diff > 0:
        direction = "down"
    elif y_diff < 0:
        direction = "up"
    elif x_diff > 0:
        direction = "right"
    elif x_diff < 0:
        direction = "left"
    else:
        direction = "none"
    return direction

def get_direction(tracks):
    current_location = {}
    current_direction = {}
    for track in tracks:
        track_id = track[1]
        if track_id not in past_location:  
            past_location[track_id] = [track[2], track[3]]
        else:
            current_location[track_id] = [track[2], track[3]]
            current_direction[track_id] = get_direction_vector(past_location[track_id], current_location[track_id])

# Update past_location after processing all tracks
    for track_id in current_location:
        past_location[track_id] = current_location[track_id]

# Clean up past_location for tracks that are no longer present
    all_present_track_ids = [track[1] for track in tracks]
    for track_id in list(past_location.keys()):
        if track_id not in all_present_track_ids:
            del past_location[track_id]

    return current_direction


# tracks is a List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.

def find_clusters_centroids_and_areas(tracks, n_clusters):
# Extract the center points of bounding boxes from each track
    points = []
    for track in tracks:
        bb_left, bb_top, bb_width, bb_height = track[2], track[3], track[4], track[5]
        center_x = bb_left + bb_width / 2
        center_y = bb_top + bb_height / 2
        points.append([center_x, center_y])

# Convert list of points to a numpy array
    points = np.array(points)

# Apply K-means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(points)
        labels = kmeans.labels_
    except:
        labels = np.zeros(len(points))

# Prepare a dictionary to store results
    clusters_info = {}

# Compute convex hull, area, and centroid for each cluster
    for i in range(n_clusters):
        cluster_points = points[labels == i]

        if len(cluster_points) < 3:
            continue  
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        centroid = np.mean(hull_points, axis=0)
        area = hull.area

        clusters_info[i] = {'centroid': centroid, 'area': area}

    return clusters_info

def get_distance_to_threshold(tracks,threshold_line_height):
    distance = {}
    for track in tracks:
        distance[track[1]] = threshold_line_height - track[3]

# get convex hull
    return distance



# CROWD DENSITY
def monitor_crowd_density(clusters_info, high_density_threshold):
    
    density_status = {}

    for cluster_id, info in clusters_info.items():
        cluster_area = info['area']
        if cluster_area >= high_density_threshold:
            status = 'High Density'
        else:
            status = 'Normal Density'
        
        density_status[cluster_id] = status

    return density_status

# FLOW DIRECTIONS
def analyze_flow_direction(directions_in_cluster):
   
    predominant_directions = {}

    for cluster_id, directions in directions_in_cluster.items():
        if directions:
            most_common_direction = max(set(directions), key=directions.count)
            predominant_directions[cluster_id] = most_common_direction
        else:
            predominant_directions[cluster_id] = 'No clear direction'

    return predominant_directions

# GET SPEED AND DIRECTION
def get_speed_and_direction(tracks, past_locations, frame_rate):
    current_locations = {}
    speeds = {}
    directions = {}

    for track in tracks:
        track_id = track[1]
        current_location = np.array([track[6], track[7]])  # x, y coordinates

        if track_id not in past_locations:
            past_locations[track_id] = [current_location]
            speed = 0
        else:
# Calculate speed
            delta_distance = np.linalg.norm(current_location - past_locations[track_id][-1])
            speed = delta_distance * frame_rate  # pixels per second

# Store current location
            past_locations[track_id].append(current_location)

# Get direction
        direction = get_direction_vector(past_locations[track_id][-1], current_location)
        directions[track_id] = direction
        speeds[track_id] = speed

# Clean up past_locations for tracks that are no longer present
    all_present_track_ids = [track[1] for track in tracks]
    for track_id in list(past_locations.keys()):
        if track_id not in all_present_track_ids:
            del past_locations[track_id]

    return speeds, directions

# STATIONARY TIME
def update_stationary_time(speeds, stationary_times, stationary_threshold=0.5):
    for track_id, speed in speeds.items():
        if speed <= stationary_threshold:
            stationary_times[track_id] += 1
        else:
            stationary_times[track_id] = 0
    return stationary_times

# ABNORMAL BEHAVIOUR
def detect_abnormal_behaviors(stationary_times, abnormal_threshold=50):
    abnormal_behaviors = []
    for track_id, stationary_time in stationary_times.items():
        if stationary_time > abnormal_threshold:
            abnormal_behaviors.append(track_id)
    return abnormal_behaviors

# CALCULATE RISK FACTOR
def calculate_risk_factor(track_id, stationary_times, abnormal_behaviors, max_stationary_time):
    risk_factor = 0

    # Increase risk factor based on stationary time
    stationary_percentage = (stationary_times[track_id] / max_stationary_time) * 100
    risk_factor += min(stationary_percentage, 50)  # Cap at 50%

    # Additional risk for abnormal behavior
    if track_id in abnormal_behaviors:
        risk_factor += 50

    return min(risk_factor, 100)  # Cap total risk at 100%

# DRAW TRACKS
def draw_tracks_2(image, tracks, directions, distances, image_shape, threshold_line_height, stationary_times, abnormal_behaviors, max_stationary_time):
    for track in tracks:
        track_id = track[1]
        bbox = track[2:6]  
        bbox_color = (0, 0, 255) if track_id in abnormal_behaviors else (0, 255, 0)  # Red for abnormal, green for normal

# Calculate risk factor
        risk_factor = calculate_risk_factor(track_id, stationary_times, abnormal_behaviors, max_stationary_time)

# Draw bounding box
        cv.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), bbox_color, 2)

# Display track ID and risk factor
        text = f"ID: {track_id}, Risk: {risk_factor:.2f}%"
        cv.putText(image, text, (int(bbox[0]), int(bbox[1] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# Additional drawing (directions, distances, etc.) as needed

    return image

### KPI SIDEBAR
def add_kpi_sidebar(image, kpi_texts, sidebar_width=200, background_color=(255, 255, 255)):
    
    height, width, _ = image.shape
    sidebar = np.full((height, sidebar_width, 3), background_color, dtype=np.uint8)

    # Position for the first line of text
    y_pos = 40

# Write each KPI text
    for text in kpi_texts:
        cv.putText(sidebar, text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 20  

# Concatenate the sidebar to the right of the image
    combined_image = np.concatenate((image, sidebar), axis=1)
    return combined_image

def detect_abandoned_luggage(bboxes, confidences, class_ids):
   
    person_id = 0
    luggage_ids = [24, 26, 28]

    person_bboxes = [bboxes[i] for i, class_id in enumerate(class_ids) if class_id == person_id]
    luggage_bboxes = [bboxes[i] for i, class_id in enumerate(class_ids) if class_id in luggage_ids]

    abandoned_luggage = []


    for luggage_bbox in luggage_bboxes:
        is_abandoned = True
        for person_bbox in person_bboxes:
            if intersects(luggage_bbox, person_bbox):
                is_abandoned = False
                break
        if is_abandoned:
            abandoned_luggage.append(luggage_bbox)

    return abandoned_luggage

def detect_garbage(bboxes, confidences, class_ids):
    person_id = 0
    garbage_ids = [39,40,41,44,45,46,47,48,49,50,51,52,53,54,55,80]
    
    person_bboxes = [bboxes[i] for i, class_id in enumerate(class_ids) if class_id == person_id]
    garbage_bboxes = [bboxes[i] for i, class_id in enumerate(class_ids) if class_id in garbage_ids]
    
    garbage = []

    for garbage_bbox in garbage_bboxes:
        is_present = True
        for person_bbox in person_bboxes:
            if intersects(garbage_bbox, person_bbox):
                is_present = False
                break
        if is_present:
            garbage.append(garbage_bbox)

    # Calculate the area of garbage bounding boxes
    garbage_areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in garbage]

    return garbage, garbage_areas


#detect weapon
def detect_weapon(bboxes, confidences, class_ids):
    
    weapon_ids = [42,43,76]
    
    weapon_bboxes = [bboxes[i] for i, class_id in enumerate(class_ids) if class_id in weapon_ids]

    return weapon_bboxes
    
def intersects(bbox1, bbox2):
   
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def main(video_path, model, tracker):

    
    known_face_encodings, known_face_names = get_face_encodings()
    print(known_face_names)

    headers = ["Time","Density", "Speeds", "Stationary Time", "Abnormal Behaviors"]
    
    with open('kpi.csv', 'w') as f:
        f.write(','.join(headers) + '\n')

    cap = cv.VideoCapture(0)
    frame_count = 0
    fps = 0
    start_time = time.time()
    past_locations = {}
    stationary_times = defaultdict(int)
    while True:
        ok, image = cap.read()

        
        rgb_frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
            name = "Unknown"
            
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = min(range(len(face_distances)), key=face_distances.__getitem__)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
        
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            if (name == "Unknown"):
                box_colour=(255,0,0)
            elif (name == "SECURITY"):
                box_colour=(0,255,0)
            else:
                box_colour=(0,0,255)
            cv.rectangle(image, (left, top), (right, bottom), box_colour, 2)
            
            cv.rectangle(image, (left, bottom - 35), (right, bottom), box_colour, cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    

        if not ok:
            print("Cannot read the video feed.")
            break
        
        frame_count += 1


        try:
            if time.time() - start_time >= 1:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
        except:
            pass

    # print("Frames per second using video.get(cv.CAP_PROP_FPS) : {0}".format(fps))
        image = cv.resize(image, (768,432))

    # draw a magenta line on the image at 75% height of the image
        threshold_line_height = int(image.shape[0] * 0.75)
        cv.line(image, (0, threshold_line_height ), (image.shape[1], threshold_line_height ), (255, 0, 255), 2)

        bboxes, confidences, class_ids = model.detect(image)

    # GET ABANDONED LUGGAGE BBOXES
        abandoned_luggage_bboxes = detect_abandoned_luggage(bboxes, confidences, class_ids)
        for bbox in abandoned_luggage_bboxes:
            x, y, w, h = bbox
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            ## text to display
            text = "Abandoned Luggage"
            cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Detect garbage
        garbage_bboxes, garbage_areas = detect_garbage(bboxes, confidences, class_ids)
        for bbox, area in zip(garbage_bboxes, garbage_areas):
            print(f"area : {garbage_areas}")
            if area > 25000:  # Check if the area is greater than 2500
                x, y, w, h = bbox
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                text = "Garbage"
                cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Detect weapon
        weapon_bboxes = detect_weapon(bboxes, confidences, class_ids)
        for bbox in weapon_bboxes:
            x, y, w, h = bbox
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 200), 2)
            text = "Weapon"
            cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
            

    
        # keep only for the persons
        bboxes = bboxes[class_ids == 0]
        confidences = confidences[class_ids == 0]
        class_ids = class_ids[class_ids == 0]
        
        tracks = tracker.update(bboxes, confidences, class_ids)
        directions = get_direction(tracks) 
        distances = get_distance_to_threshold(tracks,threshold_line_height) 
        n_clusters = 3  
        clusters_info = find_clusters_centroids_and_areas(tracks, n_clusters)
       

        # Calculate and display FPS every second
        speeds, directions = get_speed_and_direction(tracks, past_locations, fps)
        stationary_times = update_stationary_time(speeds, stationary_times)
        abnormal_behaviors = detect_abnormal_behaviors(stationary_times)

        ## if area below 500, there is a cluster of people
        clusteredPeople = False
        for cluster_id, info in clusters_info.items():
            if info['area'] < 1000:
                clusteredPeople = True
                ## draw a circle at the centroid with dynamic radius proportionate to the area
                radius = int(np.sqrt(info['area']) / 2)*10
                cv.circle(image, tuple(info['centroid'].astype(int)), radius, (0, 255, 0), 2)

        ## Find all directions of all ids in each ckuster
        directions_in_cluster = {}
        for cluster_id, info in clusters_info.items():
            directions_in_cluster[cluster_id] = []
            for track_id in directions:
                directions_in_cluster[cluster_id].append(directions[track_id])
        
        ## count the maximum occuring direction in each cluster
        max_directions_in_cluster = {}
        for cluster_id, info in clusters_info.items():
            max_directions_in_cluster[cluster_id] = max(set(directions_in_cluster[cluster_id]), key = directions_in_cluster[cluster_id].count)

        ## save in temporalClusterDirection
        if frame_count%30 == 0:
            temporalClusterDirection.append(max_directions_in_cluster)
        ## keep only last 10 frames
        if len(temporalClusterDirection) > 10:
            temporalClusterDirection.pop(0)
        
        ## if the direction of a cluster is same for 10 frames, then the cluster is stable
        stableCluster = False
        for cluster_id, info in clusters_info.items():
            count_up = 0
            count_down = 0
            count_left = 0
            count_right = 0
            count_none = 0
            try:
                for direction in temporalClusterDirection:
                    if direction[cluster_id] == 'up':
                        count_up += 1
                    elif direction[cluster_id] == 'down':
                        count_down += 1
                    elif direction[cluster_id] == 'left':
                        count_left += 1
                    elif direction[cluster_id] == 'right':
                        count_right += 1
                    else:
                        count_none += 1
            except:
                pass

            ## if the direction is same for 10 frames, then the cluster is stable
            if count_none>5:
                stableCluster = True
                print(f"Cluster {cluster_id} is stable")
                ## draw a circle at the centroid with dynamic radius proportionate to the area
                radius = int(np.sqrt(info['area']) / 2)*10
                cv.circle(image, tuple(info['centroid'].astype(int)), radius, (0, 0, 255), 2)
                ## draw the direction of the cluster
                if max_directions_in_cluster[cluster_id] == 'up':
                    cv.arrowedLine(image, tuple(info['centroid'].astype(int)), (int(info['centroid'][0]), int(info['centroid'][1]-radius)), (0, 0, 255), 2)
                elif max_directions_in_cluster[cluster_id] == 'down':
                    cv.arrowedLine(image, tuple(info['centroid'].astype(int)), (int(info['centroid'][0]), int(info['centroid'][1]+radius)), (0, 0, 255), 2)
                elif max_directions_in_cluster[cluster_id] == 'left':
                    cv.arrowedLine(image, tuple(info['centroid'].astype(int)), (int(info['centroid'][0]-radius), int(info['centroid'][1])), (0, 0, 255), 2)
                elif max_directions_in_cluster[cluster_id] == 'right':
                    cv.arrowedLine(image, tuple(info['centroid'].astype(int)), (int(info['centroid'][0]+radius), int(info['centroid'][1])), (0, 0, 255), 2)
                else:
                    pass


        ## BELOW THRESHOLD, if 5 people below threshold, then set alarm
        num_people_below_threshold = 0
        for track in tracks:
            if track[3] > threshold_line_height:
                num_people_below_threshold += 1
        if num_people_below_threshold > 5:
            cv.putText(image, "ALERT", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            
        high_density_threshold = 500
        density_status = monitor_crowd_density(clusters_info, high_density_threshold)
        print(density_status)

        ## FLOW DIRECTIONS
        predominant_directions = analyze_flow_direction(directions_in_cluster)
        print(predominant_directions)

        # Print or use KPIs as needed
        print("Speeds:", speeds)
        print("Stationary Times:", stationary_times)
        print("Abnormal Behaviors:", abnormal_behaviors)

        updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
        # updated_image = draw_tracks(updated_image, tracks,directions,distances,updated_image.shape,threshold_line_height)
        updated_image = draw_tracks_2(updated_image, tracks, directions, distances, updated_image.shape, threshold_line_height, stationary_times, abnormal_behaviors, max_stationary_time=30)
        
        # Prepare the KPI texts
        mean_density = np.mean([info['area'] for info in clusters_info.values()])
        # Aggregate all directions
        all_directions = list(directions.values())
        # Find the most common direction
        #most_common_direction = max(set(all_directions), key=all_directions.count)
        speeds = [speed for speed in speeds.values() if speed > 0]
        mean_speed = np.mean(speeds) if speeds else 0
        mean_stationary_time = np.mean(list(stationary_times.values()))
        abnormal_behaviors = len(abnormal_behaviors)

        kpi_texts = [
            f"Density: {mean_density}",
            f"Speeds: {mean_speed}",
            f"Stationary Time: {mean_stationary_time}",
            f"Abnormal Behaviors: {abnormal_behaviors}"
        ]

       
        
        curr_datetime = datetime.datetime.now()
        curr_datetime = curr_datetime.strftime("%d/%m/%Y %H:%M:%S")
        row = [curr_datetime, mean_density, mean_speed, mean_stationary_time, abnormal_behaviors]
        with open('kpi.csv', 'a') as f:
            f.write(','.join([str(x) for x in row]) + '\n')
        


        
            updated_image_with_sidebar = add_kpi_sidebar(updated_image, kpi_texts)



        
        cv.imwrite("output.jpg", updated_image_with_sidebar)

        cv.imshow("image", updated_image_with_sidebar)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(VIDEO_FILE, model, tracker)
    print("Done.")
    exit(0)