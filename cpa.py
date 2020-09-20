#!/usr/bin/env python
import cv2
import sys
import os
import time
import logging as log
#from tempimage import TempImage
from openvino.inference_engine import IECore,IENetwork, IEPlugin
import math
import paho.mqtt.client as mqtt
import json
# Conversion from pixels to meters
PIXELM =15.45



def main():
    mqttc = mqtt.Client()
    topic = 'camera_s'
    topic1= 'camera_a'
    mqttc.connect('MQTTSERVER', 1883,60)

    prob_threshold =0.5
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    ie = IECore()
    plugin = IEPlugin(device='CPU', plugin_dirs=None)

    # Model   person-vehicle-bike-detection-crossroad-0078   
    model_xml='person-vehicle-bike-detection-crossroad-0078.xml'
    model_bin = 'person-vehicle-bike-detection-crossroad-0078.bin'
    
    #Load neural network
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net, num_requests=1)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    
    cur_request_id = 0
    last_epoch = time.time()
    

    # Input Stream
    cap = cv2.VideoCapture('rtsp://camera address')
    
    #First Frame
    ret, frame = cap.read()
    
    log.info("To close the application, press 'CTRL+C'")
    
    #Initialize counters
    left=0
    right=0
    speed=0
    diff=0
    nc_a=0
    np_a=0
    numero_coches=0
    numero_personas=0
    status = 'searching_c'
    statusp = 'searching_p'
    Cx_a=[0,0,0,0,0]
    Cy_a=[0,0,0,0,0]
    t_a=[0,0,0,0,0]
    Direction=[0,0,0,0,0,0]
    Xmin_a = [0,0,0,0,0,0]
    Xmax_a = [0,0,0,0,0,0]
    t_a=[0,0,0,0,0]
    

    Cpx_a=[0,0,0,0,0]
    Cpy_a=[0,0,0,0,0]
    tp_a=[0,0,0,0,0]
    
    distance=16
    width=0
    speedp=0
    
    distancep=16
   
    timer=0
    count=0
    np=0
    nc=0
    n_frames=0
    
    inf_start = time.time()
    temp=time.time()    
    

    while cap.isOpened():

        # Get a new frame
        ret, frame = cap.read()
        
        while (not ret):
            ret, frame = cap.read()
            log.info('Lost Frame')
    
        # We get the frame resolution
        initial_w= cap.get(3)
        initial_h = cap.get(4)
        #Analyze a frame
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
                
        if ((exec_net.requests[cur_request_id].wait(-1) == 0) & (diff > 0.2)):
            # Parse detection results 
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            #Initialize counters of number of cars and people
            np=0
            nc=0
            i=0
            j=0
            
            #Analyze response looking for objects
            for obj in res[0][0]:
                # We check if there are objects with a probability of a threshold, and if these objects are people or cars and save the number of them
                if obj[2] > prob_threshold:
                    class_id = int(obj[1])             
                    if (class_id == 1.0):
                        np=np+1
                    elif (class_id == 2.0):
                        nc=nc+1
            
            # If we don´t detect cars we continue in searching_c state or change from tracking_c state to search_c state
            if ((status == 'tracking_c') & (nc == 0)):
                status = 'searching_c'
                Cx_a=[0,0,0,0,0]
                Cy_a=[0,0,0,0,0]
                t_a=[0,0,0,0,0]
                Direction=[0,0,0,0,0,0]
                Xmin_a = [0,0,0,0,0,0]
                Xmax_a = [0,0,0,0,0,0]
                t_a=[0,0,0,0,0]
                nc_a=0

            #n_frames is the number of frames we wait to chenge to search state. With persons in some frames cn be missed and we count another time in the next frame he appers
            if ((statusp == 'tracking_p')& (np ==0)):
                if  (n_frames >100):
                    n_frames=10
                else:
                    n_frames=n_frames +1
                
                
            
            # If we don´t detect people we continue in searching_p state or change from tracking_p state to search_c state            
            if ((statusp == 'tracking_p') & (np == 0) & (n_frames >10)):
                statusp = 'searching_p'
                Cpx_a=[0,0,0,0,0]
                Cpy_a=[0,0,0,0,0]
                tp_a=[0,0,0,0,0]
                
           
            #If we are tracking buttherei another person we change to search state
            if ((statusp=='tracking_p') & (np> np_a) & (n_frames >10)):
                j=np_a
                statusp='searching_p'
                n_frames=0

            
            #Analyse the objects
            for obj in res[0][0]:
                if (obj[2] > prob_threshold):
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    class_id = int(obj[1])
                    
                    #Cars
                    if (class_id == 2.0):
                        #In searching_c state we calculate the centroid and create a new member of the list for each car detected and update the total cars number and the number of cars going
                        #to the right or to the left. If we have a park on the left we can check the availabity using a rest betwenn these two data

                        if (status == 'searching_c'):
                            numero_coches=numero_coches + nc
                            
                            Xmin_a[i]=xmin
                            Xmax_a[i]=xmax
                            #Centroid
                            Cx_a[i]=(xmin + xmax)/2
                            Cy_a[i]=(ymin+ ymax)/2
                            #Counter right and left
                            if (((ymin + ymax)/2) <300):
                                left=left +1
                                Direction[i]=0                        
                            else:
                                right=right+1
                                Direction[i]=1
                            
                            # Draw black rectangle around the car
                            p1=(xmin,ymin)
                            p2=(xmax,ymax)                         
                            cv2.rectangle(frame, p1, p2, (0,0,0), 2, 1)
                            
                            # If there are two cars in the same direction we can calculate the distance between them if the go in the same direction
                            # The direction is easy to check using the Cy parameter
                            # The y coordate is almost constant so we can use only x coordinate
                            if ((nc> 1) & ((i-1) >= 0)):
                                if ( Direction[i] == Direction [i-1]):
                                        if  (Direction[i] == 0):
                                            distance = (PIXELM/initial_w)*(Xmax_a[i] - Xmin_a[i-1])
                                        else:
                                            distance = (PIXELM/initial_w)*(Xmin_a[i] - Xmax_a[i])
                            else:
                                distance=16
                            
                            #We save the number of cars detected
                            nc_a =nc
                            #update the time
                            t_a[i]=time.time()
                            n_frames=0
                            
                            #We change state when we have the data of all the cars in frame
                            if ((i+1) >= nc):
                                status='tracking_c'
                            

                        elif (status == 'tracking_c'):
                            # In tracking mode we update the parameter of each car in the frame and compare with the data in the frame before
                            #In the borders there are more error so we avoid these measures
                            if ((xmin > 30) & (xmax< w -30)):
                                
                                #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                                
                                # We estimate width checking xmax and xmin and using a factor of m/pixels
                                p1=(xmin,0,0)
                                p2=(xmax,0,0)
                                width = (PIXELM/initial_w)*math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1,p2)]))

                                #we calculate the centroid
                                Cx1=(xmin + xmax)/2
                                Cy1=(ymin + ymax)/2   
                                
                                #Speed
                                #Car goes to the left. The objects are detected from left to right
                                #If there are two cars in the first frame there are detected first the first car then the second
                                # but in the next frame the second car is detected first
                                #Car goes to the right. The objects are detected from left to right
                                #If there are two cars in the first frame there are detected first the first car then the second
                                #in the next frame the same first the first car second the second car
                                    
                                if (Cy1<300):
                                    p1= (Cx_a[nc_a-i],Cy_a[nc_a-i],0)
                                    t= time.time() - t_a[nc_a-i] 
                                
                                else:
                                    p1= (Cx_a[i],Cy_a[i],0)
                                    t= time.time() - t_a[i] 

                                p2=(Cx1,Cy1,0)
                                d = (PIXELM/initial_w)*math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1,p2)]))
                                
                                #With the time and the distance the speed = distance / time
                                if (t != 0):
                                    speed= (d/t)*3.6
                                
                                #if we detect more of one car in the same direction we can check the distance between them in order to know if the second is a safety distance
                                if ((nc> 1) & ((i-1) >= 0)):
                                    if ( Direction[i] == Direction [i-1]):
                                            if  (Direction[i] == 0):
                                                distance = (PIXELM/initial_w)*(Xmax_a[i] - Xmin_a[i-1])
                                            else:
                                                distance = (PIXELM/initial_w)*(Xmin_a[i] - Xmax_a[i])
                                else:
                                    distance=16
                                
                                p1=(xmin,ymin)
                                p2=(xmax,ymax)
                                if ((distance <9) or (speed >30)):
                                    cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
                                else:
                                    cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)    
                                t=time.time()
                                
                                
                                #Update the variables in the right position for the next tracking check

                                if (Cy1<300):
                                    Cx_a[nc_a-i]=Cx1
                                    Cy_a[nc_a-i]=Cy1
                                    t_a[nc_a-i]=time.time() 
                                else:
                                    Cx_a[i]=Cx1
                                    Cy_a[i]=Cy1
                                    t_a[i] =time.time()
                                n_frames=0
                                nc_a=nc
                               
                        i = i+1
                    
                    elif (class_id == 1.0):
                        if (statusp == 'searching_p'):
                            if ( ymin > 20):
                                numero_personas=numero_personas + np
                                np = np-np_a
                                Cpx_a[j]=(xmin + xmax)/2
                                Cpy_a[j]=(ymin+ ymax)/2
                                
                                # Draw black rectangle
                                p1=(xmin,ymin)
                                p2=(xmax,ymax)                         
                                cv2.rectangle(frame, p1, p2, (0,0,0), 2, 1)
                                
                                # If there are more than one people we can calculate the distance between them 
                                if ((np> 1)&(j>=1)):
                                    p1=(Cpx_a[j],Cpy_a[j],0)
                                    p2=(Cpx_a[j-1],Cpy_a[j-1],0)
                                    d= (PIXELM/initial_w)*math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1,p2)]))
                                    if (d !=0):
                                        distancep = d    

                                else:
                                    distancep=15
                                
                                np_a =np
                                tp_a[j]=time.time()
                                if ((j+1) >= np):
                                    statusp='tracking_p'
                            

                        elif (statusp == 'tracking_p'):
                            if ( ymin > 80):
                                Cx1=(xmin + xmax)/2
                                Cy1=(ymin + ymax)/2
                                #Speed of persons
                                
                                p1= (Cpx_a[j],Cpy_a[j],0) #Position same person frame before
                                p2=(Cx1,Cy1,0)# Position in this frame
                                d = (PIXELM/initial_w)*math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1,p2)]))
                                t= time.time() - tp_a[j] 

                                if (t != 0):
                                    speedp= (d/t)*3.6

                                if ((np> 1) & ((np_a-j)<len(Cpx_a))):
                                    p2=(Cpx_a[np_a-j],Cpy_a[np_a-j],0) #Position other person frame before
                                    d= (PIXELM/initial_w)*math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1,p2)])) 
                                    if (d !=0):
                                        distancep = d 
                                else:
                                    distancep=15
                                
                                p1=(xmin,ymin)
                                p2=(xmax,ymax)
                                if (distancep <2):
                                    cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
                                else:
                                    cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

                                np_a=np
                                Cpx_a[j]=Cx1
                                Cpy_a[j]=Cy1
                                tp_a[j]=time.time()
                              
                        j = j+1


            last_epoch=time.time()
             
        nc=0
        np=0
        diff=(time.time()-last_epoch)
               
        
        counter_cars = "Car Number: {}".format(left+right)
        left_cars = "Left Cars: {}".format(left)
        right_cars = "Right Cars: {}".format(right)
        
        velocidad_message = "Speed: {:.1f} kmh".format(speed)
        d_message = "Distance: {:.1f} m".format(distance)
        w_message = "Width: {:.1f} m".format(width)
        
        # We construct the mqtt message

        
        #we send it every minute   
        timer = time.time()- temp
        if (timer > 60):
            data = {
            "nc": left+right,
            "left": left,
            'right':right,
            'people': numero_personas
            }

            payload = json.dumps(data)
            mqttc.publish(topic,payload)
            temp=time.time()
            log.info('envio mqtt')

            

        # We blur the top part, my neightbour´s house
        
        area= frame[0:180, 0: 1280]
        area = cv2.blur(area, (25,25), 0)
        frame[0:180, 0: 1280]=area
        


        counter_people = "People Number: {}".format(numero_personas)
        s_people= "Speed: {:.1f} kmh".format(speedp)
        d_people = "Distance: {:.1f} m".format(distancep)

        
        cv2.putText(frame, counter_cars, (15, 25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, left_cars, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, right_cars, (15, 75), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, w_message, (15, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)     
        
        cv2.putText(frame, counter_people, (int (initial_w-300), 25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, s_people, (int (initial_w-300), 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        
        #Rules Check
        
        #Vehicle Speed
        if (speed < 30):
            cv2.putText(frame, velocidad_message, (15, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)            
        else:
            cv2.putText(frame, velocidad_message, (15, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), 2)        
            cv2.imwrite("car_speed%d.jpg" % count, frame)
            data = 'Speed car violation a picture has been saved'
            payload = json.dumps(data)
            mqttc.publish(topic1,payload)
            count=count+1
            speed=0
        
        #Security Distance
        if (distance > 9):
            cv2.putText(frame, d_message, (15, 190), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)        
        else:
            cv2.putText(frame, d_message, (15, 190), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite("car_nodistancee%d.jpg" % count, frame)
            data = 'Car distance violation a picture has been saved'
            payload = json.dumps(data)
            mqttc.publish(topic1,payload)
            count=count+1
            distance=16

        #Social distance        
        if (distancep <2) :
            cv2.putText(frame, d_people, (int (initial_w-300), 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite("pepple_nodistancee%d.jpg" % count, frame)
            log.info('Image Saved people_nodistance') 
            data = 'Social distance violation a picture has been saved'
            payload = json.dumps(data)
            mqttc.publish(topic1,payload)
            count=count+1
            distancep=16
        else:
            cv2.putText(frame, d_people, (int (initial_w-300), 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255, 0), 2)
            distance=16
        

        #Show the image with the information
        cv2.imshow('Results', frame)
            
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    sys.exit(main() or 0)
