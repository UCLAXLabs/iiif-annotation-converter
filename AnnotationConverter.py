#!/usr/bin/python # Expects Python 3
import json
import os
import requests
from slugify import slugify
import pickle
from PIL import Image
from io import BytesIO
from lxml import etree

# Given a IIIF annotation "activity stream" (basically a manifest with annotation data),
# - request all of the annotated images, RESIZED to the desired dimensions for maching
# learning (this is usually not more than 1000 pixels on the shortest side) and save them
# as JPEGs or whatever
# - output the annotation bounding boxes and tags, with the bounding boxes resized at the
# same ratios as the full image (see above), in an appropriate format for machine learning
# work -- 

projectName = 'edo_illustrations'

# Keys: manifest URL. Values: dictionary of key: canvas ID, value: image data from manifest
maniMappings = {}
# Keys: imageID. Values: resized width and height
trainingImages = {}

# Keys: imageID. Values: array of dictionaries of annotations with tags and bounding boxes
imageAnnotations = {}

allTags = set()

cacheEnabled = True
cachePath = os.path.join(os.getcwd(), 'cache/')

# Where the resized JPEG images are stored
imagesFolder = os.path.join(os.getcwd(), 'images/')
# Where the XML image files are stored
annotationsFolder = os.path.join(os.getcwd(), 'annotations/')
xmlsFolder = os.path.join(annotationsFolder, 'xmls/')

try:
  if (not os.path.exists(imagesFolder)):
    os.makedirs(imagesFolder)
except:
  print("Unable to create output images folder, exiting")
  import sys
  sys.exit()

try:
  if (not os.path.exists(annotationsFolder)):
    os.makedirs(annotationsFolder)
    os.makedirs(xmlsFolder)
except:
  print("Unable to create annotations folders, exiting")
  import sys
  sys.exit()

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
    data = requests.get(link)
    if (useCache):
      pickle.dump(data, open(filePath, 'wb'))
  return data

def processManifest(maniData):
  theseMappings = {} 
  for sequence in maniData['sequences']:
    for canvas in sequence['canvases']:
      canvasID = canvas['@id']
      for image in canvas['images']:
        #imageURL = image['resource']['@id']
        theseMappings[canvasID] = image
  return theseMappings

#jsonPath = "/Users/broadwell/Dropbox/Library/HYU_MA/marinus_annotations_9-7.json"
#jsonFile = open(jsonPath, 'r')
#annotData = json.load(jsonFile)

annotationURL = "http://marinus.library.ucla.edu/viewer/annotation/"
annotData = getURL(annotationURL).json()

for r in annotData['resources']:
  rID = r['@id']
  rType = r['@type']
  if (rType != 'oa:Annotation'):
    continue
  tags = []
  for ann in r['resource']:
    # If it has "@type" : "dctypes:Text" then this is a transcription
    # If it has "@type" : "oa:Tag" then this is a tag -- deal with it
    if (ann['@type'] == 'oa:Tag'):
      val = ann['chars']
      if ((val != "") and (val not in tags)):
        tags.append(val)

  region = r['on'][0]
  srcManifest = region['within']['@id'] # this is usually "@type" : "sc:Manifest"

  if (srcManifest not in maniMappings):
    manifestData = getURL(srcManifest).json()
    newMappings = processManifest(manifestData)
    maniMappings[srcManifest] = newMappings

  bbox = region['selector']['default']['value'] # xywh=X,Y,W,H
  xywh = list(map(int, bbox.split('=')[1].split(',')))
  # The 'full' field looks like this: "http://marinus.library.ucla.edu/images/kabuki/canvas/ucla_bib1987273_no005_rs_001.tif.json"
  # Usually it doesn't resolve to anything -- it's usually just a canvas (?) ID
  canvasID = region['full']
  imageID = canvasID.split('/')[-1].replace('.json','').replace('.tif','').replace('.png','').replace('.jpg','') + ".jpg"
  
  if imageID not in imageAnnotations:
    imageAnnotations[imageID] = []

  # http://marinus.library.ucla.edu/loris/kabuki/ucla_bib1987273_no005_rs_001.tif/full/full/0/default.jpg
  fullURL = maniMappings[srcManifest][canvasID]['resource']['@id']
  fullWidth = maniMappings[srcManifest][canvasID]['resource']['width']
  fullHeight = maniMappings[srcManifest][canvasID]['resource']['height']
      
  if (imageID not in trainingImages):
    resizedURL = fullURL.replace('full/full','full/!1000,1000')
    imageResponse = getURL(resizedURL)
    im = Image.open(BytesIO(imageResponse.content))
    resizedWidth, resizedHeight = im.size
    
    outputPath = os.path.join(imagesFolder, imageID)

    with open(outputPath, 'w') as outputFile:
      print("saving cropped image " + imageID)
      im.save(outputFile, 'jpeg')
      
    trainingImages[imageID] = (resizedWidth, resizedHeight)

  else:
    resizedWidth, resizedHeight = trainingImages[imageID]

  # Format of cropBox: xmin, ymin, xmax, ymax
  cropBox = (xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3])

  widthRatio = float(resizedWidth) / float(fullWidth)
  heightRatio = float(resizedHeight) / float(fullHeight)

  resizedCropBox = (int(float(cropBox[0]) * widthRatio),
                    int(float(cropBox[1]) * heightRatio),
                    int(float(cropBox[2]) * widthRatio),
                    int(float(cropBox[3]) * heightRatio))

  thisAnnotation = {'tags': tags, 'bbox': resizedCropBox} 
  imageAnnotations[imageID].append(thisAnnotation)

  # This is how to format a request for the full-res contents of the bbox
  #croppedImage = im.crop(cropBox)
  #croppedImageID = imageID + '.' + ",".join(list(map(str, cropBox))) + '.jpg'

for imageID in imageAnnotations:
  xmlID = imageID.replace('.jpg', '').replace('.png', '').replace('.tif', '')
  print("writing XML annotation file " + xmlID)
  root = etree.Element("annotation")
  folder = etree.SubElement(root, "folder")
  folder.text = projectName
  fn = etree.SubElement(root, "filename")
  fn.text = xmlID
  sz = etree.SubElement(root, "size")
  wd = etree.SubElement(sz, "width")
  wd.text = str(resizedWidth)
  ht = etree.SubElement(sz, "height")
  ht.text = str(resizedHeight)
  dp = etree.SubElement(sz, "depth")
  dp.text = "3" # No alpha channel, watch out for grayscale
  seg = etree.SubElement(root, "segmented")
  seg.text = "0"
  for anno in imageAnnotations[imageID]:
    obj = etree.SubElement(root, "object")
    name = etree.SubElement(obj, "name") # This is the tag
    if ("figure" in anno["tags"]):
      thisTag = "figure"
    elif ("animal" in anno["tags"]):
      thisTag = "animal"
    else:
      thisTag = anno["tags"][0]
    name.text = thisTag
    allTags.add(thisTag)
    bbox = etree.SubElement(obj, "bndbox")
    xmi = etree.SubElement(bbox, "xmin")
    xmi.text = str(anno['bbox'][0])
    ymi = etree.SubElement(bbox, "ymin")
    ymi.text = str(anno['bbox'][1])
    xma = etree.SubElement(bbox, "xmax")
    xma.text = str(anno['bbox'][2])
    yma = etree.SubElement(bbox, "ymax")
    yma.text = str(anno['bbox'][3])
  
  xmlPath = os.path.join(xmlsFolder, xmlID + '.xml')
  with open(xmlPath, 'wb') as xmlFile:
    et = etree.ElementTree(root)
    et.write(xmlFile, pretty_print=True)

trainingListFile = os.path.join(annotationsFolder, 'trainval.txt')
with open(trainingListFile, 'w') as trainingList:
  for imageID in trainingImages:
    baseID = imageID.replace('.png', '').replace('.jpg', '').replace('.tif', '')
    trainingList.write(baseID + "\n")

labelMapFile = os.path.join(os.getcwd(), 'label_map.pbtxt')
with open(labelMapFile, 'w') as labelMap:
  tagCount = 1
  for tag in allTags:
    labelMap.write("item {\n id: " + str(tagCount) + "\n name: '" + tag + "'\n display_name: '" + tag + "'\n}\n")
    tagCount += 1
