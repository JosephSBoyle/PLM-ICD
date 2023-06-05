"""Adapted from copyrighted code owned by Canon. All rights reserved."""
from numpy import exp
from typing import Sequence

def sigmoid(x):
    return 1 / (1 + exp(-x))

WORD_RGB = ('103', '13', '103')

def html_header(title):
    return f"<!DOCTYPE html>\n<html lang=\"en\" dir=\"ltr\">\n<head>\n<meta charset=\"utf-8\">\n<title>{title}</title>\n</head>\n<body>\n<div style=\"font-family:'Segoe UI'\">"

HTML_FOOTER = '</div>\n</body>\n</html>'


def html_space(pixels=10):
    return f"<span style=\"display: inline-block; width: {pixels}px\"></span>"


def get_rgba(rgb, alpha=0):
    return f"rgba({','.join(rgb)}, {alpha})"

def word_overlay(word, w_att, s_att=0):
    output_string = ' '
    color = 'white' if w_att > 0.5 or s_att > 0.5 else 'black'
    output_string += f"<span style='background-color:{get_rgba(WORD_RGB, w_att)}; color:{color}'>{word}</span>"
    return output_string

def overlay_word_attention(sentence, attention_word):
    output_string = ""

    # Iterate over words
    for word, w_att in zip(sentence, attention_word):
        output_string += word_overlay(word, w_att)

    output_string += html_space()
    output_string += '\n</p>\n</span>'
    return output_string

def overlay_sentence_attention(text: str, attention_word, name: str, true_label: str, prediction: float, label_description: str = None) -> str:
    # Open file
    output = ""
        # Write title and true vs predicted labels
    output += html_header(name) \
        + f"<p><b>Ground truth: </b>{true_label}</p>\n" \
        + f"<p><b>Prediction:   </b>{'positive' if prediction > 0.5 else 'negative'}</p>\n" \
        + f"<p><b>p̂(positive | text): </b>{sigmoid(prediction):.2f}</p>\n" \
        
    if label_description:
        output += f"<p><b>Label description:</b> <i>{label_description} </i>" + "\n"
        

    word_bar = "<span style='border:solid black 1px'>"
    for i in range(0, 11):
        i /= 10
        color = 'white' if i > 0.5 else 'black'
        word_bar += f"<span style='padding: 0 0.5em; background-color:{get_rgba(WORD_RGB, i)}; color:{color}'>{i}</span>"
    word_bar += "</span>"
    output += f"\ngradient: {word_bar}\n</p>\n"

    # Write sentence
    output += f"<h2></h2>\n" \
        + overlay_word_attention(text, attention_word) \
        + HTML_FOOTER
    
    return output
