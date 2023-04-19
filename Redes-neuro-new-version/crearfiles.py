import numpy as np
import cv2
import os

numeros_prueba = range(1,2)

for number in numeros_prueba:
    #imgreaded = f'./Proc-img/Redes-neuro-new-version/{number}grilla.jpg'
    #img = cv2.imread(imgreaded, cv2.IMREAD_GRAYSCALE)
    countAdd = 0
    countTry = 0
    for i in range(0,280*10, 28):

        ini_ver = i 
        fin_ver = i+27
        #print("i",i)
        countAdd = 0
        for j in range(0, 280, 28):
            ini_hor = j
            fin_hor = j+27
            find = False
            #print("j",j)
            
            while(find == False):
                imgToAddRead = f"./archive/trainingSet/trainingSet/{number}/img_{countTry}.jpg"

                if os.path.exists(imgToAddRead):
                    imgToAdd = cv2.imread(imgToAddRead, 0)
                    imgToAdd = cv2.resize(imgToAdd, (28, 28))
                    imgToAdd = cv2.bitwise_not(imgToAdd)
                    countAdd+= 1
                    countTry+= 1
                    find = True
                    #print(f'ADD: img_{countTry}')
                    break
                if countTry>100000:
                    break
                else:
                    countTry+= 1    
                    #print(countTry)
                if find == False:
                    countTry+=1
                    #print(countTry)                    

            if countAdd <= 1:
                resulth = imgToAdd.copy()
            else: 
                toAddh = resulth.copy()
                resulth = cv2.hconcat([toAddh, imgToAdd])
        
        #cv2.imwrite(f'N{i}.jpg',resulth)

        if i == 0:
            resultv = resulth.copy()
        else:
            toAddV = resulth.copy()
            tempv = resultv.copy()
            resultv = cv2.vconcat([tempv, toAddV])
        #print(countAdd)
    
    viejaread = f'./Proc-img/Redes-neuro-new-version/{number}-1grilla.jpg'
    viejaAdd = cv2.imread(viejaread, 0)
    resultv = cv2.vconcat([viejaAdd, resultv])
    cv2.imwrite(f"grilla{number}.jpg",resultv)


