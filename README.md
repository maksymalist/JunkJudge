# JunkJudge README

JunkJudge is an AI-powered web application that helps users classify waste materials as either recyclable, non-recyclable or hazardous. This document serves as a guide for users on how to use JunkJudge.

## Getting Started

To use JunkJudge, visit the [website](https://www.junkjudge.com/) and upload an image of the waste material that you want to classify. JunkJudge will then use its AI algorithms to analyze the image and classify the waste material into one of the following categories:

- Recyclable
- Non-recyclable
- Biological

## Features

- JunkJudge's AI algorithms are constantly learning and improving, which means that the accuracy of its classifications will continue to improve over time.
- JunkJudge provides users with recommendations on how to properly dispose of waste materials based on their classification.
- JunkJudge's user interface is intuitive and easy to use, making it accessible to a wide range of users.
- JunkJudge is constantly updating its database of waste materials and their classifications to ensure that it remains accurate and up-to-date.

## Limitations

- JunkJudge is not 100% accurate and may misclassify certain waste materials.
- JunkJudge is currently only able to classify waste materials based on images, which means that it cannot classify waste materials that are not visible in images (e.g. gases, liquids).
- JunkJudge's recommendations on how to dispose of waste materials may not be applicable in all regions or countries.

## Dive into the algorithm 🤖 🧠 …

### Description

> JunkJudge's AI algorithm leverages the power of Ensemble Learning, with which it combines the outputs of 2 Convolutional Neural Network that were trained using transfer learning with the “ConvNext-tiny” dataset called “Neo” and “Trinity”. To get the final verdict, it passes both of the predictions trough the final model called “Morpheus”
> 

### “Neo”

“Neo” is a codename given to the first model which takes in an image as an input and spits out a probability distributed over 7 classes (biological 🌱,  trash 🗑️,  paper 📄,  cardboard 📦,  metal 🤘, glass 🍾,  plastic ♳). 

### “Trinity”

“Trinity” is a codename given to the first model which takes in an image as an input and spits out a probability distributed over 6 classes (trash 🗑️,  paper 📄,  cardboard 📦,  metal 🤘, glass 🍾,  plastic ♳).  Notice that Trinity does not have a biological output, the reason for that being that Neo is capable enough to judge for it’s self that an item is biological waist given their unique features.

### “Morpheus”

“Morpheus” is like the glue that holds everything together. It uses a custom made dataset called “Oracle”, that was made using the combination of the predicated probabilities of “Neo” and “Trinity”, to classify the item using the given stats.
