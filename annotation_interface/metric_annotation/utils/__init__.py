import os
import random

import streamlit.components.v1 as components

build_dir = "utils/highlighter/frontend/build"
_component = components.declare_component("annotation_tools", path=build_dir)


def highlight_text(article, phrases, color):
    for phrase in phrases:
        if phrase in article:
            article = article.replace(
                phrase, f'<span style="background-color: #{color};">{phrase}</span>'
            )
    return article