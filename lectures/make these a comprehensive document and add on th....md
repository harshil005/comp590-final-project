# **COMP 590: Information Visualization – Comprehensive Course Guide**

**Source:** Consolidated analysis of Lecture Slides (Internal) and Course Bookmarks (External).  
This document serves as the master reference for the course, organizing all materials into five core modules: **Foundations**, **Perception**, **Methodology**, **Interaction**, and **Advanced Applications**.

## **I. Introduction & The Tool Landscape**

### **1\. What is Visualization?**

* **Definition:** The use of computer-supported, interactive, visual representations of abstract data to amplify cognition.  
* **Purpose:** We visualize to find structure in data that statistics might miss (e.g., *Anscombe's Quartet*), to handle large volumes of data, and to tell stories.

### **2\. The Ecosystem of Tools**

There is no "perfect" tool; the choice depends on the trade-off between **Expressivity** (control), **Learnability**, and **Efficiency**.

| Tool Type | Examples | Pros | Cons |
| :---- | :---- | :---- | :---- |
| **Manual** | Illustrator, Pen & Paper | Total creative freedom. Great for prototyping. | Not scalable. Tedious to update. |
| **WYSIWYG** | **Tableau**, PowerBI, Excel | Fast, easy to learn. Automated query-to-vis loop. | Limited flexibility. Hard to create novel designs. |
| **Grammar-Based** | **Vega-Lite**, **Altair** | Declarative syntax (JSON/Python). Automates interaction. | Learning curve. |
| **Programmatic** | **D3.js**, Matplotlib | Maximum control. Scalable. | Steepest learning curve. Requires coding. |
| **Hybrid/Layout** | **Charticulator** | Constraint-based layout for custom glyphs without code. | Niche use cases. |

* **External Resource:** [Vega-Lite: A Grammar of Interactive Graphics](https://idl.cs.washington.edu/files/2017-VegaLite-InfoVis.pdf) — Introduces the "Grammar of Interaction," treating user inputs as data streams.

## **II. Foundations: Data & Perception**

### **3\. Data Abstraction**

Before designing, you must define *what* you are visualizing.

* **Data Types:** Tabular (rows/cols), Networks (nodes/links), Fields (continuous), Geometry (spatial).  
* **Attribute Types:**  
  * **Categorical:** Identity (Apple vs. Orange).  
  * **Ordered:** Ordinal (Small, Medium, Large) or Quantitative (10cm, 20cm).  
* **Visual Building Blocks:**  
  * **Marks:** The geometric primitive (Point, Line, Area).  
  * **Channels:** How we modify the mark (Position, Size, Color, Shape, Tilt).

### **4\. The Human Visual System**

* **Processing Modes:**  
  * **Fast (Parallel):** The brain instantly extracts "broad statistics" (averages, clusters) and "pop-out" features (red dot in a sea of blue).  
  * **Slow (Serial):** Detailed comparisons (e.g., "Is bar A exactly 2x taller than bar B?") require conscious, serial attention.  
* **Gestalt Principles:** We perceive groups, not just points.  
  * *Proximity & Similarity:* Things close together or looking alike are grouped.  
  * *Enclosure & Connection:* Boundaries and lines create strong groups.  
* **Subitizing:** We can instantly count only **\~4-6 items**.  
  * *Design Rule:* Categorical color palettes should not exceed 6 colors, or users will struggle to distinguish them.  
* **Weber’s Law:** We perceive differences proportionally. It is harder to distinguish 100 vs 101 than 1 vs 2\.  
* **Change Blindness:** Users often miss changes in a dashboard unless animated transitions guide their attention.

### **5\. Color Theory**

* **Color Spaces:**  
  * **sRGB:** Computer standard (not perceptually uniform).  
  * **CIELAB:** Perceptually uniform (math distance \= visual distance).  
  * **HSL/HSV:** Artist-friendly.  
* **Encodings:**  
  * **Sequential:** For continuous/ordered data. Must have monotonic lightness (Light $\\to$ Dark).  
  * **Diverging:** For data with a neutral midpoint (Negative $\\to$ Zero $\\to$ Positive).  
  * **Categorical:** For distinct groups. Colors must be equidistantly salient.  
* **The "Rainbow" Rule:** **Avoid the rainbow map.** It lacks intuitive ordering and creates artificial boundaries.  
* **External Tools:**  
  * [**Colorgorical**](http://vrl.cs.brown.edu/color/pdf/colorgorical.pdf?v=5dd92af6d1e6c5584236275adc769e82)**:** Generates palettes by balancing perceptual distance with semantic name difference.  
  * [**Color Crafting**](https://arxiv.org/abs/1908.00629)**:** Automates "designer quality" ramps.

### **6\. Misleading Visualization**

* **The Lie Factor (Tufte):** $\\frac{\\text{Visual Effect}}{\\text{Data Effect}}$. Ideally \~1.0.  
* **Common Misleaders:**  
  * **Truncated Axes:** Bar charts starting at non-zero values exaggerate differences.  
  * **Area/Volume:** Using radius to represent magnitude exaggerates the value quadratically.  
  * **Cherry-Picking:** Hiding data to force a trend.  
  * **Dark Patterns:** Design choices that intentionally confuse (e.g., cumulative graphs hiding a decline).

## **III. Design Methodology & Evaluation**

### **7\. Task Analysis**

You cannot build a tool without knowing the user's specific goals.

* **Goal (Why):** Explore, Confirm, Present, or Enjoy.  
* **Means (How):** Navigation, Organizing, Relation.  
* **Targets (What):** Trends, Outliers, Topology, Shape.  
* **Shneiderman's Mantra:** *"Overview first, zoom and filter, then details-on-demand."*

### **8\. Design Frameworks**

* **Five Design Sheets (FDS):** A rapid, low-fidelity sketching method (Brainstorm $\\to$ 3 Iterations $\\to$ Realization). Best for early stages.  
* **Design Studies Methodology (DSM):** A rigorous 9-stage process for real-world problem solving.  
  * *Stages:* Learn $\\to$ Winnow $\\to$ Cast $\\to$ Discover $\\to$ Design $\\to$ Implement $\\to$ Deploy $\\to$ Reflect $\\to$ Write.  
  * *Key Paper:* [Design Study Methodology: Reflections from the Trenches](https://www.google.com/search?q=https://ieeexplore.ieee.org/abstract/document/6327248).

### **9\. Evaluation**

* **Formative:** Done *during* design (e.g., Think-alouds, Sketches).  
* **Summative:** Done *after* to measure success.  
* **Insight-Based Evaluation:** Counting the number/depth of insights a user generates, rather than just task time.  
* **Implicit Error:** [Framework](https://sci.utah.edu/~vdl/papers/2018_infovis_IE-Framework.pdf) for visualizing the "known unknowns" and expert uncertainties that usually don't make it into the dataset.

## **IV. Interaction**

### **10\. The Role of Interaction**

Static charts handle small data; interaction handles scale.

* **Taxonomy of Intent (Yi et al.):**  
  1. **Select:** Mark items.  
  2. **Explore:** Pan/Show me something else.  
  3. **Reconfigure:** Sort/Rearrange.  
  4. **Encode:** Change chart type.  
  5. **Abstract/Elaborate:** Tooltips/Zoom.  
  6. **Filter:** Remove data.  
  7. **Connect:** Brushing & Linking (highlighting connections).  
* **Cost of Interaction:** Interaction incurs cognitive load and time costs. It should not be the default if a static view suffices.  
* **Next-Gen Interaction:**  
  * **Eviza:** A natural language interface ("Show me sales in March") that bridges the gap between dashboards and conversation.

## **V. Advanced Applications**

### **11\. Data Storytelling**

* **Narrative Structure:**  
  * **Author-Driven:** Linear, heavy messaging (Video).  
  * **Reader-Driven:** Free exploration, no messaging (Dashboard).  
  * **Martini Glass:** The gold standard—start with a tight author-driven narrative (the stem), then open up to reader-driven exploration (the bowl).  
* **Reference:** [Narrative Visualization: Telling Stories with Data](http://vis.stanford.edu/files/2010-Narrative-InfoVis.pdf).

### **12\. Data Physicalization**

* **Definition:** Encoding data in physical artifacts (3D prints, sculptures).  
* **Benefits:** Active perception (touch), accessibility, and emotional engagement.  
* **Dynamic Physicalization:** Using **Zooids** (micro-robots) to create physical scatterplots that can update in real-time.  
* **Autographic Vis:** Data that records itself (tree rings, wear patterns).

### **13\. Immersive Analytics**

* **Definition:** Using VR/AR to place the analyst *inside* the data.  
* **Paradigms:**  
  * **Data Worlds:** Purely virtual spaces for abstract data.  
  * **Situated Analytics:** Overlaying data onto real-world objects (e.g., patient stats floating over a patient).  
* **Grand Challenges:** Precise interaction in mid-air, collaboration in VR, and integrating ubiquitous analytics into daily life.