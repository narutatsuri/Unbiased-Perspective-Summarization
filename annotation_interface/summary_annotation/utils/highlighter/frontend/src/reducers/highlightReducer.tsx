import React from "react"
import { IState, IAction, ActionTypes } from "../types/highlightTypes"
import { Streamlit } from "streamlit-component-lib"

// Define the initial state of the component
export const initialState: IState = {
  text: "",
  actual_text: [],
  highlights: [],
  nhighlights: [],
  selectedReference: "P0",
}

// Reducer function to handle state transitions
export const reducer = (state: IState, action: IAction): IState => {
  switch (action.type) {
    case ActionTypes.SET_TEXT_HIGHLIGHTS:
      // Set the text andhighlight
      Streamlit.setComponentValue([action.payload.highlights, action.payload.nhighlights])
      return {
        ...state,
        text: action.payload.text,
        highlights: action.payload.highlights,
        nhighlights: action.payload.nhighlights,
      }

    case ActionTypes.RENDER_TEXT:
      // Logic to render text withhighlight
      const { text, highlights, nhighlights } = state
      const actual_text: JSX.Element[] = []
      let start = 0
      let selectedReference = state.selectedReference
      let label = selectedReference.charAt(0)
      let index = Number(selectedReference.slice(1))

      if (label === "N") {
        if (!nhighlights[index]) {
          index = 0
          selectedReference = "N0"
          if (!nhighlights[index]) {
            nhighlights[index] = []
          }
        }
      } else {
        if (!highlights[index]) {
          index = 0
          selectedReference = "P0"
          if (!highlights[index]) {
            highlights[index] = []
          }
        }
      }



      if (label == "N") {
        nhighlights[index]
          ?.sort((a, b) => a.start - b.start)
          .forEach((highlight, index) => {
            actual_text.push(
              <span key={`unhighlighted-${index}`}>
                {text.substring(start, highlight.start)}
              </span>
            )
            actual_text.push(
              <span
                key={`nhighlighted-${index}`}
                className="nhighlighted border border-nprimary bg-nprimary20"
              >
                {text.substring(highlight.start, highlight.end)}
              </span>
            )
            start = highlight.end
          })
          actual_text.push(
            <span key="nhighlighted-end">{text.substring(start)}</span>
          )
        } else {
          highlights[index]
          ?.sort((a, b) => a.start - b.start)
          .forEach((highlight, index) => {
            actual_text.push(
              <span key={`unhighlighted-${index}`}>
                {text.substring(start, highlight.start)}
              </span>
            )
            actual_text.push(
              <span
                key={`highlighted-${index}`}
                className="highlighted border border-primary bg-primary/20"
              >
                {text.substring(highlight.start, highlight.end)}
              </span>
            )
            start = highlight.end
          })
          actual_text.push(
            <span key="highlighted-end">{text.substring(start)}</span>
          )
        }

      

      Streamlit.setComponentValue([highlights, nhighlights])
      return {
        ...state,
        actual_text,
        selectedReference,
      }

    case ActionTypes.ADD_REFERENCE:
      // Add a new reference
      let add_label = action.payload
      const newHighlights = [...state.highlights]
      const newNHighlights = [...state.nhighlights]
      
      let newIndex = "P" + newHighlights.length
      if (add_label === "N") {
        newIndex = "N" + newNHighlights.length
        newNHighlights.push([])
      } else {
        newHighlights.push([])
      }

      return {
        ...state,
        highlights: newHighlights,
        nhighlights: newNHighlights,
        selectedReference: newIndex,
      }

    case ActionTypes.SELECT_REFERENCE:
      return {
        ...state,
        selectedReference: action.payload[0] + action.payload[1],
      }

    case ActionTypes.REMOVE_REFERENCE:
      const rem_label = action.payload[0]
      const rem_index = action.payload[1]
      if (rem_label === "N") {
        const updatedNHighlights = [...state.nhighlights]
        updatedNHighlights.splice(rem_index, 1)
        return {
          ...state,
          highlights: [...state.highlights],
          nhighlights: updatedNHighlights,
          selectedReference: "N" + 0,
        }
      } else {
        const updatedHighlights = [...state.highlights]
        updatedHighlights.splice(rem_index, 1)
        return {
          ...state,
          highlights: updatedHighlights,
          nhighlights: [...state.nhighlights],
          selectedReference: "P" + 0,
        }
      }

    default:
      return state
  }
}
