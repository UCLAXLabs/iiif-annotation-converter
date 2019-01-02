#!/usr/bin/python # Expects Python 3
import json
import os
import requests
from slugify import slugify
import pickle
from PIL import Image
from io import BytesIO
from lxml import etree

# Given links to one or more IIIF annotation "activity streamw" (basically lists of annotations 
# in manifest-like format):
#
# 1) request all of the annotated images, RESIZED to the desired dimensions for maching
# learning (this is usually not more than 1000 pixels on the shortest side) and save them
# as JPEGs or whatever
#
# 2) output the annotation bounding boxes and tags, with the bounding boxes resized at the
# same ratios as the full image (see above), in appropriate formats to train machine
# learning models. This project uses the Tensorflow object detection libraries, so the
# necessary outputs are as follows:
#
# label_map.pbtxt -- a mapping of label numbers to names (can be linked data URIs) and display_names
# annotations/trainval.txt -- a list of the downloaded/resized image filenames *without extensions*
# annotaions/xmls/*.xml -- one XML file per annotated image, in PASCAL VOC annotation format

projectName = 'edo_illustrations'

# Can include more than one source of annotations here
annotationURLs = ["https://marinus.library.ucla.edu/viewer/annotation/"]

#allowedManifests = None
allowedManifests = ['https://marinus.library.ucla.edu/iiif/kabuki/manifest.json', 'https://marinus.library.ucla.edu/iiif/gokan/manifest.json']

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

clipImages = False
outputFolder = os.path.join(os.getcwd(), 'output/')
if (clipImages):
  try:
    if (not os.path.exists(outputFolder)):
      os.makedirs(outputFolder)
  except:
    print("Unable to create clippings folder, won't save them")
    clipImages = False

# Set to True to save resized versions of *all* images in any collection
# manifests referenced in the annotations manifests -- convenient if
# you're planning to run a trained model against all possible images
# locally (without re-fetching the images from their IIIF servers)
harvestAll = False

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
  if (not os.path.exists(xmlsFolder)):
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
        fullURL = image['resource']['@id']
        imageID = canvasID.split('/')[-1].replace('.json','').replace('.tif','').replace('.png','').replace('.jpg','') + ".jpg"
  
        # If it's in trainingImages, it has already been downloaded
        if (harvestAll and (imageID not in trainingImages)):
          outputPath = os.path.join(imagesFolder, imageID)
          if (not os.path.isfile(outputPath)):
            resizedURL = fullURL.replace('full/full','full/!1000,1000')
            imageResponse = getURL(resizedURL)
            im = Image.open(BytesIO(imageResponse.content))
    
            with open(outputPath, 'w') as outputFile:
              print("harvesting cropped image " + imageID)
              im.save(outputFile, 'jpeg')

        theseMappings[canvasID] = image
  return theseMappings

# Map original tags onto a smaller domain
def reduceTags(tagList):
  if ("figure" in tagList):
    thisTag = "figure"
  elif ("animal" in tagList):
    thisTag = "animal"
  else:
    thisTag = tagList[0]
  return thisTag

def normalizeTag(tag):
  if ((tag == 'samrai') or (tag == 'samuari')):
    tag = 'samurai'
  elif (tag == 'stading'):
    tag = 'standing'
  return tag

#jsonPath = "/Users/broadwell/Dropbox/Library/HYU_MA/marinus_annotations_9-7.json"
#jsonFile = open(jsonPath, 'r')
#annotData = json.load(jsonFile)

for annotationURL in annotationURLs:
  print("Fetching annotations from",annotationURL)
  annotData = getURL(annotationURL).json()

  for r in annotData['resources']:
    rID = r['@id']
    rType = r['@type']
    if (rType != 'oa:Annotation'):
      continue
   
    # XXX Technically the annotation can be instantiated on more than one image source.
    # Not sure how often this will happen. For now we just use the first one
    region = r['on'][0]
    if ('@id' in region['within']):
      srcManifest = region['within']['@id'] # this is usually "@type" : "sc:Manifest"
    else:
      srcManifest = region['within']

    if ((allowedManifests is not None) and (srcManifest not in allowedManifests)):
      print("Annotation source not in list of allowed manifests, skipping:", srcManifest)
      continue

    # Download the full source manifest if we haven't seen it before
    if (srcManifest not in maniMappings):
      manifestData = getURL(srcManifest).json()
      newMappings = processManifest(manifestData)
      maniMappings[srcManifest] = newMappings

    tags = []
    for ann in r['resource']:
      # If it has "@type" : "dctypes:Text" then this is a transcription
      # If it has "@type" : "oa:Tag" then this is a tag -- deal with it
      if (ann['@type'] == 'oa:Tag'):
        val = ann['chars']
        if ((val != "") and (val not in tags)):
          tags.append(val)

    bbox = region['selector']['default']['value'] # xywh=X,Y,W,H
    xywhString = bbox.split('=')[1]
    xywh = list(map(int, xywhString.split(',')))
    # The 'full' field looks like this: "http://marinus.library.ucla.edu/images/kabuki/canvas/ucla_bib1987273_no005_rs_001.tif.json"
    # Often it doesn't resolve to anything -- it's typically just a canvas (?) ID
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

    # This saves all of the training regions in the 'output/' folder
    if (clipImages):
      thisTag = reduceTags(tags)
      # This is how to format a request for the full-res contents of the bbox
      croppedURL = fullURL.replace('full/full', xywhString + '/full')
      # NOTE: The version below resizes the cropped image to 1000 pixels on its
      # longest side; it does not seem to be possible to resize the entire image 
      # first and THEN crop it (?) using IIIF
      #croppedURL = resizedURL.replace('full/!1000,1000', xywhString + '/!1000,1000')
      croppedResponse = getURL(croppedURL)
      # We could also grab the full image and crop it (or resize and then crop 
      # it), but it's better to make the image server do the work
      #croppedImage = im.crop(cropBox)
    
      croppedImage = Image.open(BytesIO(croppedResponse.content))
      croppedImageID = imageID + '.' + xywhString + '_' + thisTag + '.jpg'
      #resizedWidth, resizedHeight = croppedImage.size
    
      croppedPath = os.path.join(outputFolder, croppedImageID)

      with open(croppedPath, 'w') as croppedFile:
        print("saving cropped image " + croppedImageID)
        croppedImage.save(croppedFile, 'jpeg')

# Generate the output files
for imageID in imageAnnotations:
  resizedWidth, resizedHeight = trainingImages[imageID]
  xmlID = imageID.replace('.jpg', '').replace('.png', '').replace('.tif', '')
  print("writing XML annotation file " + xmlID)
  root = etree.Element("annotation")
  folder = etree.SubElement(root, "folder")
  folder.text = projectName
  fn = etree.SubElement(root, "filename")
  fn.text = xmlID + '.jpg'
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
    for tag in anno["tags"]:
      tag = normalizeTag(tag)
      allTags.add(tag)
      obj = etree.SubElement(root, "object")
      name = etree.SubElement(obj, "name") # This is the tag
      name.text = tag
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
