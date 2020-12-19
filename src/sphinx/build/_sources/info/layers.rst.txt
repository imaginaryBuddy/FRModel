######
Layers
######

The project will host 3 levels. Separated so we can focus on achieving goals independently without overlap.

==============
Back (Layer 0)
==============

Essentially anything in the `frmodel` package, will house the python OOP implementation,
opening only important front-end functions to those who will be meddling with the data.

Making messing with the data in Jupyter-Notebook accessible without having to rewrite multiple
boilercode mess.

Goal: Make sure that **Layer 1** doesn't write any OOP, only functional code.

================
Middle (Layer 1)
================

The middle here is mainly pointing to users, like data scientists, focus more on the theoretical
knowledge in implementing clear-cut instructions instead of having to do low-end algorithmic
code.

Goal: Make sure that **Layer 2** only needs minimal code.

===============
Front (Layer 2)
===============

This is where all the UI, front-facing implementation will be done.

There are multiple choices to pivot off of:

- PyQt (Platform independent Application UI)
- Plotly (Web based UI, standalone website building)
- RShiny (R Lang with Python Binding)

Goal: Make sure that the **end-user** can understand the data