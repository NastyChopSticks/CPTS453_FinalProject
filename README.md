# WSU Graph-Based GPS Navigation System

## Overview
This project is a final assignment for a Graph Theory course, where students were given full creative freedom in designing a graph-based application.

This project implements a **WSU campus navigation system** that models over **500 campus locations** as a weighted graph. It uses **Dijkstra’s Algorithm** to compute the most optimal path between two selected points.

Although this was a class assignment, it was a personal project I was highly motivated to build and refine.

---

## Features
- Graph-based representation of WSU campus
- Over 500 mapped locations
- Shortest path calculation using Dijkstra’s Algorithm
- Manual path navigation between selected start and end points
- Interactive visualization using a web interface

---

## Technologies Used
- Python
- :contentReference[oaicite:0]{index=0} (UI)
- :contentReference[oaicite:1]{index=1} (geospatial data handling)
- :contentReference[oaicite:2]{index=2} (map visualization)
- :contentReference[oaicite:3]{index=3} (manual map/path creation)

---

## How It Works
- Campus locations are represented as graph nodes
- Walkable paths between locations are edges with weights
- Dijkstra’s Algorithm computes the shortest path between two nodes
- The result is displayed on an interactive map interface

---

## Limitations
- No real-time GPS tracking
- Users must manually select start and end locations
- Some campus paths may be missing or approximated due to manual mapping limitations

---

## Future Improvements
- Add real-time GPS tracking for live navigation
- Convert into a mobile application
- Improve path accuracy by physically surveying campus routes
- Optimize or refine graph connectivity for better route accuracy

---

## Installation

### 1. Install dependencies
```bash id="3kq9nd"
pip install streamlit geopandas leafmap
```

### 2. Clone the repository
```bash id="v8wqpa"
git clone <your-repo-url>
cd <repo-folder>
```

### 3. Run the application
```bash id="m1x7zc"
streamlit run main.py
```

---

## Reflection
This project was an opportunity to apply graph theory concepts in a real-world setting. The most challenging aspect was not implementation of algorithms, but accurately modeling the campus using geospatial tools. It provided valuable experience in graph modeling, geospatial data handling, and application design.
