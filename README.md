# Circular-Objects-Detection

## The most important things- in 10 bullet points:

1. This repository depicts an attempt to create an algorithm that detects circular objects in images.

2. The main goal of this algorithm is to be a quick preprocessing algorithm for autonomous vehicles. If there is an algorithm that can detect traffic lights with high accuracy but it has a relatively long running time, the repository’s algorithm can be useful (in principle).
 
3. The repository’s algorithm is relatively fast, and if it detects a circular object in the image, then the slower and more accurate algorithm can be executed.

4. The slower algorithm can be executed such that it processes first the objects that the preprocessing algorithm detects and the area around them, and only then it processes the rest of the image (if it’s necessary for the application). Therefore, using this preprocessing algorithm may reduce the time it takes to detect traffic lights.

5. The accuracy of the repository’s algorithm is 55.55%.

6. The running time is much faster in comparison to any algorithm that uses CannyEdges or HoughCircles transform.

7. In principle, this algorithm can be useful, but in practice, for images of streets in cities there is a 40% chance that the algorithm will classify an object as a circle, although it isn’t.

8. However, this probability is significantly lower for images of roads outside of cities.

9. To sum up: this algorithm may be useful for roads outside of cities, as a preprocessing algorithm that detects circular objects in the image, and in particular traffic lights, with a moderate success rate but with a relatively short running time.

10. The full information about the algorithm appears in the PDF file in this repository.
