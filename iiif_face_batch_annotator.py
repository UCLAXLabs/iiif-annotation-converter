import numpy as np
import os
import sys
import json
import uuid

import face_recognition

from io import StringIO
from PIL import Image

from io import BytesIO
import pickle
import requests
from slugify import slugify

from time import sleep

# XXX THIS CODE DOESN'T WORK WELL ON RALPH DUE TO MULTIPROCESSING ISSUES
# IN THE SOURCE CODE (PROBABLY OF dlib) FOR MULTIPLE CPUS

sys.path.append("..")

# Set to True to create a IIIF curation file (manifest) of the detected faces
writeCuration = True
# Set to True to create a IIIF annotation file of the detected faces
writeAnnotations = True

# Number of times to upsample the image looking for faces. Higher numbers find
# smaller faces.
# Upsample values greater than 2 exceed available VRAM
UPSAMPLE = 2
BATCH_SIZE = 16

# Set to True to write images of detected faces to a folder
saveImages = True

cacheEnabled = True
cachePath = os.path.join(os.getcwd(), 'cache/')

#targetManifests = []
#with open('manifest_list.txt', 'r') as manifestDoc:
#  for maniLine in manifestDoc:
#    targetManifests.append(maniLine.strip())
targetManifests = ['https://marinus.library.ucla.edu/iiif/ucla_ua/manifest.json']
#'http://marinus.library.ucla.edu/images/kabuki/manifest.json',
#'http://marinus.library.ucla.edu/images/gokan/manifest.json', ]
# Could add more manifests from Keio's ukiyo-e collection, especially Hiroshige
# http://dcollections.lib.keio.ac.jp/en/ukiyoe/4

outputFolder = os.path.join(os.getcwd(), 'output/')
if (saveImages):
  try:
    if (not os.path.exists(outputFolder)):
      os.makedirs(outputFolder)
  except:
    print("Unable to create clippings folder, won't save them")
    saveImages = False

# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_detection_on_single_image(image):
  #image = face_recognition.load_image_file(imagePath)
  face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=UPSAMPLE, model="cnn")
  #for face_location in face_locations:
  #  top, right, bottom, left = face_location
  return face_locations

def run_detection_on_multiple_images(imageArray):
  batchSize = len(imageArray)
  locations_array = face_recognition.api.batch_face_locations(imageArray, number_of_times_to_upsample=UPSAMPLE, batch_size=batchSize)
  return locations_array

try:
  if (not os.path.exists(cachePath)):
    os.makedirs(cachePath)
except:
  print("Unable to create cache folder, will download everything")
  cacheEnaled = False

def getURL(link, useCache=cacheEnabled):
  fileSlug = slugify(link)
  filePath = os.path.join(cachePath, fileSlug)
  if (useCache and os.path.isfile(filePath)):
    data = pickle.load(open(filePath, 'rb'))
    print("fetched from cache: " + link)
  else:
    print("fetching " + link)
    data = requests.get(link, verify=False)
    if (useCache):
      pickle.dump(data, open(filePath, 'wb'))
  return data

def processManifest(maniData):
  manifestLabel = maniData['label']
  theseMappings = {}
  for sequence in maniData['sequences']:
    for canvas in sequence['canvases']:
      canvasID = canvas['@id']
      for image in canvas['images']:

        # Could explicitly save the files here, but they'll already be in the cache,
        # so why bother?
        #with open(outputPath, 'w') as outputFile:
        #  print("harvesting cropped image " + imageID)
        #  im.save(outputFile, 'jpeg')
        if (canvasID not in theseMappings):
          theseMappings[canvasID] = [image]
        else:
          theseMappings[canvasID].append(image)
  return manifestLabel, theseMappings

def processBatch(imageBatch, batchMetadata):

  global isFirstAnnotation, isFirstBox, isFirstRange, annoFile, curationFile

  iHeight = imageBatch[0].shape[0]
  iWidth = imageBatch[0].shape[1]
  iDims = str(iWidth) + ':' + str(iWidth)

  sizeOfBatch = len(batchMetadata)
  print("Processing batch of size",sizeOfBatch,"dimensions",iDims)

  if (sizeOfBatch == 1):
    faces_array = [run_detection_on_single_image(imageBatch)]
  else:
    faces_array = run_detection_on_multiple_images(imageBatch)

  print("Faces array is of size",len(faces_array))

  for i in range(0,len(faces_array)):

    imageMetadata = batchMetadata[i]
    imageID = imageMetadata['id']
    image_height = imageMetadata['height']
    image_width = imageMetadata['width']

    widthRatio = imageMetadata['width_ratio']
    heightRatio = imageMetadata['height_ratio']

    srcManifest = imageMetadata['src_manifest']
    canvasID = imageMetadata['canvas_id']

    # PMB Maybe also check whether class label is in a desired subset?
    print("Detected faces set for",imageID,"is of size",len(faces_array[i]))

    for face in faces_array[i]:

      print("face points are", face) # format: top, right, bottom, left
                                   # (ymin, xmax, ymax, xmin)
      face_width = float(face[1]) - float(face[3])
      face_unit_width = face_width / image_width
      print("face unit width",face_unit_width)
      face_height = float(face[2]) - float(face[0])
      face_unit_height = face_height / image_height
      print("face unit height",face_unit_height)
      face_proportion = face_unit_width * face_unit_height
      print("proportion is",face_proportion)

      #if (face_proportion < min_face_proportion_thresh):
      #  continue

      if (face_proportion < .01):
        proportionString = "<.01"
      else:
        proportionString = str(round(face_proportion,2))

      #face_width = face_unit_width * image_width
      print("face_width",face_width)
      #face_height = face_unit_height * image_height
      print("face_height",face_height)

      # face detection format: top, right, bottom, left
      #                        (ymax, xmax, ymin, xmin)
      # But XYWH wants xmin, ymax, width, height
      face_X = float(face[3])
      #face_X = float(face[1]) * image_width
      print("face_X",face_X)
      face_Y = float(face[0])
      #face_Y = float(face[0]) * image_height
      print("face_Y",face_Y)

      xywh = list(map(int, (face_X, face_Y, face_width, face_height)))
      xywhString = ','.join(list(map(str,xywh)))
      print("face xywh on resized image is",xywhString)
 
      fullXYWH = list(map(int, (face_X * widthRatio, face_Y * heightRatio, face_width * widthRatio, face_height * heightRatio)))
      fullXYWHstring = ','.join(list(map(str,fullXYWH)))
      print("face xywh on full-sized image is",fullXYWHstring)

      if (saveImages):
        #croppedImageID = imageID + '.' + xywhString + '.png'
        # Use the full XYWH string to help with debugging
        croppedImageID = imageID + '.' + fullXYWHstring + '.png'
        # Format of cropBox: xmin, ymin, xmax, ymax
        cropBox = (xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3])
        croppedImage = im.crop(cropBox)
        croppedPath = os.path.join(outputFolder, croppedImageID).replace('%7C', '_')

        #with open(croppedPath, 'w') as croppedFile:
        print("saving cropped image " + croppedImageID)
        croppedImage.save(croppedPath, 'png') 

      if (writeAnnotations):
        svgUUID = str(uuid.uuid4())

        pathTopLeft = [ str(float(fullXYWH[0])), str(float(fullXYWH[1])) ]
        pathHalfWidth = str(float(fullXYWH[2]) / 2)
        pathHalfHeight = str(float(fullXYWH[3]) / 2)

        # SVG paths are formatted
        # e.g., d="M1557.25466,1770.37267h-492.25466v0h-492.25466v310.32919v310.32919h492.25466h492.25466v-310.32919z"
        # M startX, startY (top right corner)
        # h -1/2 X h -1/2 X (draw to left in .5 w increments -- n)o idea why)
        # v 1/2 Y v 1/2 Y (draw down in .5 h increments)
        # h 1/2 X h 1/2 X (draw up in .5 w increments)
        # v -1/2 Y (draw up .5 h)
        # z (draw the rest of the way back to M -- WHY DO IT THIS WAY???)

        svgPath = "M" + pathTopLeft[0] + "," + pathTopLeft[1] + 'h' + pathHalfWidth + 'h' + pathHalfWidth + 'v' + pathHalfHeight + 'v' + pathHalfHeight + 'h-' + pathHalfWidth + 'h-' + pathHalfWidth + 'v-' + pathHalfHeight + 'z'

        svgString = "<svg xmlns='http://www.w3.org/2000/svg'>" + '<path xmlns="http://www.w3.org/2000/svg" d="' + svgPath + '" data-paper-data="{&quot;strokeWidth&quot;:1,&quot;rotation&quot;:0,&quot;deleteIcon&quot;:null,&quot;rotationIcon&quot;:null,&quot;group&quot;:null,&quot;editable&quot;:true,&quot;annotation&quot;:null}" id="rectangle_' + svgUUID + '" fill-opacity="0" fill="#00bfff" fill-rule="nonzero" stroke="#00bfff" stroke-width="1" stroke-linecap="butt" stroke-linejoin="miter" stroke-miterlimit="10" stroke-dasharray="" stroke-dashoffset="0" font-family="none" font-weight="none" font-size="none" text-anchor="none" style="mix-blend-mode: normal"/></svg>'

        # Note: DO NOT add #xywh=coords to 'full' -- the annotation import
        # will fail.
        boxJSON = {
          '@type': "oa:Annotation",
          'motivation': [ "oa:commenting", "oa:tagging" ],
          "resource": [ { '@id': "_:b2", '@type': "dctypes:Text", 'http://dev.llgc.org.uk/sas/full_text': "", 'format': "text/html", 'chars': "" },
                        { '@id': "_:b3", '@type': "oa:Tag", 'http://dev.llgc.org.uk/sas/full_text': "face", 'chars': "face" } ],
          "on": [ { '@id': "_:b0", '@type': "oa:SpecificResource", 
                    'within': { '@id': srcManifest,
                                '@type': "sc:Manifest" },
                    'selector': { '@id': "_:b1", '@type': "oa:Choice", 'default': { '@id': "_:b4", '@type': "oa:FragmentSelector", 'value': "xywh=" + fullXYWHstring }, 'item': { '@id': "_:b5", '@type': "oa:SvgSelector", 'value': svgString } }, 'full': canvasID} ], 
          "@context": "http://iiif.io/api/presentation/2/context.json" }

        box_str = json.dumps(boxJSON)
        if (isFirstAnnotation == True):
          annoFile.write(box_str + "\n")
          isFirstAnnotation = False
        else:
          annoFile.write("," + box_str + "\n")

      if (writeCuration):
        boxJSON = {
          "@id": canvasID + "#xywh=" + fullXYWHstring,
          "type": "sc:Canvas",
          "label": imageID,
          "metadata": [
            {
              "label": "tag",
              "value": "face" 
            },
            #{
            #  "label": "confidence",
            #  "value": pct_score
            #},
            {
              "label": "proportion",
              "value": proportionString
            }
          ]
        }

        box_str = json.dumps(boxJSON)
        if (isFirstBox == True):
          curationFile.write(box_str + "\n")
          isFirstBox = False
        else:
          curationFile.write("," + box_str + "\n")

# MAIN
  
iiifProject = 'ucla_ua'
iiifDomain = 'https://marinus.library.ucla.edu/iiif/' + iiifProject

annoFile = None
curationFile = None

if (writeAnnotations):
  annoFile = open('face_annotations.json', 'w')
  annoFile.write('{"@id": "https://marinus.library.ucla.edu/viewer/annotation/", "@context": "http://iiif.io/api/presentation/2/context.json", "@type": "sc:AnnotationList", "resources": [ ')

if (writeCuration):
  curationFile = open('face_curation.json', 'w')
  curationUUID = str(uuid.uuid4())
  curationFile.write('{ "@context": [ "http://iiif.io/api/presentation/2/context.json", "http://codh.rois.ac.jp/iiif/curation/1/context.json" ], "@type": "cr:Curation", "@id": "' + iiifDomain + '/json/' + curationUUID + '", "label": "Curation list", "selections": [ ')

isFirstAnnotation = True
isFirstRange = True

# This is where relevant data from each manifest will be stored
maniMappings = {}
manifestLabels = {}

# Parse each of the specified manifests
for srcManifest in targetManifests:
  if (srcManifest not in maniMappings):
    print("Processing manifest",srcManifest)
    manifestData = getURL(srcManifest).json()
    maniLabel, newMappings = processManifest(manifestData)
    manifestLabels[srcManifest] = maniLabel
    maniMappings[srcManifest] = newMappings

imageBatchMetadata = {}
imageBatches = {}

for srcManifest in maniMappings:
  maniLabel = manifestLabels[srcManifest]
  isFirstBox = True
  if (writeCuration):

    if (not isFirstRange):
      curationFile.write(', ')
    else:
      isFirstRange = False

    curationFile.write('{ "@id": "' + srcManifest + '/range/r1", "@type": "sc:Range", "label": "Objects detected by Python face_recognition package", "members": [')

  for canvasID in maniMappings[srcManifest]:
    for image in maniMappings[srcManifest][canvasID]: 

      fullURL = image['resource']['@id']
      # IIIF presentation always returns a .jpg
      #imageID = canvasID.split('/')[-1].replace('.json','').replace('.tif','').replace('.png','').replace('.jpg','').replace('.jpeg','') + ".jpg"
      imageID = image['resource']['service']['@id'].split('/')[-1].replace('.tif','').replace('.png','').replace('.jpg','').replace('.jpeg','') + ".jpg"

      imageInfoURL = image['resource']['service']['@id'] + '/info.json'

      try:
        imageInfo = getURL(imageInfoURL).json()
        fullWidth = imageInfo['width']
        fullHeight = imageInfo['height']
      except:
        # Sometimes this manifest lies, so it's better to get the info directly
        # from the image server (without actually downloading the full image)
        print("ERROR getting image info, reading dimensions from manifest")
        fullWidth = image['resource']['width']
        fullHeight = image['resource']['height']
      print("FULL IMAGE WH",fullWidth,fullHeight)

      # SOME manifests (cough, Keio Unversity, cough) don't include the
      # /full/full/0/default.jpg
      # after the image filename, so correct this...
      if (fullURL.find('/full/full/0/default.jpg') == -1):
        fullURL = image['resource']['service']['@id']
        if (fullURL.find('/full/full/0/default.jpg') == -1):
          fullerURL = fullURL + '/full/full/0/default.jpg'
          print("WARNING: irregular image ID",fullURL,"expanding to",fullerURL)
          fullURL = fullerURL

      # Files will be cached as byte streams by default
      resizedURL = fullURL.replace('full/full','full/!1000,1000')
      try:
        imageResponse = getURL(resizedURL)
      except:
        sleep(1)
        try:
          imageResponse = getURL(resizedURL)
        except:
          print("error getting image " + resizedURL + ", skipping")
          sleep(10)
          continue

      im = Image.open(BytesIO(imageResponse.content)).convert('RGB')
      resizedWidth, resizedHeight = im.size

      #image = Image.open(image_path).convert('RGB')
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.

      image_np = load_image_into_numpy_array(im)
      # These should be the same as resizedWidth and resizedHeight
      image_height = image_np.shape[0]
      image_width = image_np.shape[1]
      image_area = image_height * image_width

      image_dims = str(image_width) + ":" + str(image_height)

      if (image_dims not in imageBatches):
        imageBatches[image_dims] = []
        imageBatchMetadata[image_dims] = []

      imageBatches[image_dims].append(image_np)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      #image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.

      # Compute the ratio between resized and full-sized image
      # Multiple the resized (smaller) size by this factor to get
      # the desired value for the full-sized image
      heightRatio = fullHeight / image_height
      widthRatio = fullWidth / image_width

      #print("image area is",image_area)
      #print("image width is",image_width)
      #print("image height is",image_height)

      imageMetadata = {'id': imageID, 'full_url': fullURL, 'info_url': imageInfoURL, 'width': image_width, 'height': image_height, 'full_width': fullWidth, 'full_height': fullHeight, 'height_ratio': heightRatio, 'width_ratio': widthRatio, 'src_manifest': srcManifest, 'canvas_id': canvasID }
      imageBatchMetadata[image_dims].append(imageMetadata)

      # Run batch processing on any collection of images (binned by dimensions)
      # that exceeds the batch length threshold
      for batchDims in imageBatchMetadata:
        if (len(imageBatchMetadata[batchDims]) >= BATCH_SIZE):
          processBatch(imageBatches[batchDims], imageBatchMetadata[batchDims])
          imageBatches[batchDims] = []
          imageBatchMetadata[batchDims] = []

      #print("running detection on " + imageID)
      #faces = run_detection_on_single_image(image_np)

      # If a detected object satisfies certain conditions (size, class, score),
      # add it to the JSON curation document.
 
      # Check model confidence??? 

      #boxes = output_dict['detection_boxes']
      #classes = output_dict['detection_classes']
      #scores = output_dict['detection_scores']

  # Process the final sets of images from the manifest
  for batchDims in imageBatchMetadata:
    if (len(imageBatchMetadata[batchDims]) > 0):
      processBatch(imageBatches[batchDims], imageBatchMetadata[batchDims])
      imageBatches[batchDims] = []
      imageBatchMetadata[batchDims] = []

  if (writeCuration):
    curationFile.write('], "within": { "@id": "' + srcManifest  + '", "@type": "sc:Manifest", "label": "' + iiifProject + '" } }')

if (writeAnnotations):
  annoFile.write(' ] }')
  annoFile.close()

if (writeCuration):
  # Do this at the very end
  curationFile.write(' ] }')
  curationFile.close()
