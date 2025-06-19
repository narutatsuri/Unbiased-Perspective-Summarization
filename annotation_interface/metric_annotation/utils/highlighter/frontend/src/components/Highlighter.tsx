import React, { useReducer, useEffect, useCallback } from "react"
import { Streamlit } from "streamlit-component-lib"
import { useRenderData } from "../utils/StreamlitProvider"
import { initialState, reducer } from "../reducers/highlightReducer"
import {
  isHighlighted,
  removeHighlight,
  getCharactersCountUntilNode,
  adjustSelectionBounds,
} from "../helpers/highlightHelpers"
import { IState, ActionTypes, IAction } from "../types/highlightTypes"

const Highlighter: React.FC = () => {
  const { args } = useRenderData()
  const [state, dispatch] = useReducer<React.Reducer<IState, IAction>>(
    reducer,
    initialState
  )

  useEffect(() => {
    const fetchData = async () => {
      const { text, highlights, nhighlights } = args
      dispatch({
        type: ActionTypes.SET_TEXT_HIGHLIGHTS,
        payload: { text, highlights, nhighlights },
      })
      dispatch({ type: ActionTypes.RENDER_TEXT })
      Streamlit.setComponentValue([highlights, nhighlights])
    }

    fetchData()
  }, [])

  useEffect(() => {
    dispatch({ type: ActionTypes.RENDER_TEXT })
  }, [state.highlights, state.nhighlights, state.selectedReference])

  const handleMouseUp = useCallback(async () => {
    const selection = document.getSelection()?.getRangeAt(0)

    if (selection && selection.toString().trim() !== "") {
      const container = document.getElementById("actual-text")
      const charsBeforeStart = getCharactersCountUntilNode(
        selection.startContainer,
        container
      )
      const charsBeforeEnd = getCharactersCountUntilNode(
        selection.endContainer,
        container
      )

      const finalStartIndex = selection.startOffset + charsBeforeStart
      const finalEndIndex = selection.endOffset + charsBeforeEnd

      const textContent = container?.textContent || ""

      const { start, end } = adjustSelectionBounds(
        textContent,
        finalStartIndex,
        finalEndIndex
      )
      const selectedText = textContent.slice(start, end)

      const label = state.selectedReference.charAt(0)
      const index = Number(state.selectedReference.slice(1))
      let selectedHighlight = state.highlights[index]
      if (label === "N") {
       selectedHighlight = state.nhighlights[index]
      }

      if (
        isHighlighted(
          finalStartIndex,
          finalEndIndex,
          selectedHighlight,
        )
      ) {
        const highlights = removeHighlight(
          start,
          end,
          selectedHighlight,
        )
        if (label === "N") {
          const newNHighlights = [...state.nhighlights]
          newNHighlights[index] = highlights
          dispatch({
            type: ActionTypes.SET_TEXT_HIGHLIGHTS,
            payload: { text: state.text, highlights: [...state.highlights], nhighlights: newNHighlights },
          })
        } else {
          const newHighlights = [...state.highlights]
          newHighlights[index] = highlights
          dispatch({
            type: ActionTypes.SET_TEXT_HIGHLIGHTS,
            payload: { text: state.text, highlights: newHighlights, nhighlights: [...state.nhighlights] },
          })
        }

      } else {
        const newHighlight = { start, end, label: selectedText }
        if (label === "N") {
          const newNHighlights = [...state.nhighlights]
          newNHighlights[index] = [
            ...newNHighlights[index],
            newHighlight,
          ]
          dispatch({
            type: ActionTypes.SET_TEXT_HIGHLIGHTS,
            payload: { text: state.text, highlights: [...state.highlights], nhighlights: newNHighlights },
          })
        } else {
          const newHighlights = [...state.highlights]
          newHighlights[index] = [
            ...newHighlights[index],
            newHighlight,
          ]
          dispatch({
            type: ActionTypes.SET_TEXT_HIGHLIGHTS,
            payload: { text: state.text, highlights: newHighlights, nhighlights: [...state.nhighlights] },
          })
        }
      }
    }
  }, [state, dispatch])

  const addReference = (label: string) => {
    dispatch({ type: ActionTypes.ADD_REFERENCE, payload: label })
  }

  const selectReference = (index: string) => {
    const label = index.charAt(0)
    const hindex = Number(index.slice(1))
    dispatch({ type: ActionTypes.SELECT_REFERENCE, payload: [label, hindex] })
  }

  const removeReference = (index: string) => {
    const label = index.charAt(0)
    const hindex = Number(index.slice(1))
    dispatch({ type: ActionTypes.REMOVE_REFERENCE, payload: [label, hindex] })
  }
 

  return (
    <div>
      {/* ----- Positive Highlights ----- */}
      <div className="flex flex-row flex-wrap">
        <div
          className="flex flex-wrap justify-between items-center cursor-pointer py-1 px-3 mr-2 mb-2 rounded text-white text-base bg-primary hover:bg-secondary"
          onClick={() => addReference("P")}
        >
          <span>+ Key Point</span>
        </div>
        {state.highlights.map((reference, pindex) => (
          <span
            key={`Evidence-P${pindex}`}
            className={
              "flex flex-wrap justify-between items-center cursor-pointer py-1 px-3 mr-2 mb-2 rounded text-base" +
              (state.selectedReference === "P" + pindex
                ? " bg-primary hover:bg-secondary text-white"
                : " border border-primary text-primary hover:bg-primary hover:text-white")
            }
            onClick={() => selectReference("P" + pindex)}
          >
            P{pindex+1}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 ml-3 hover:text-gray-300"
              viewBox="0 0 20 20"
              fill="currentColor"
              onClick={() => removeReference("P" + pindex)}
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              />
            </svg>
          </span>
        ))}
      </div>
      {/* ----- Negative Highlights ----- */}
      <div className="flex flex-row flex-wrap">
        <div
          className="flex flex-wrap justify-between items-center cursor-pointer py-1 px-3 mr-2 mb-2 rounded text-white text-base bg-nprimary hover:bg-nsecondary"
          onClick={() => addReference("N")}
        >
          <span>+ Missing Key Point</span>
        </div>
        {state.nhighlights.map((reference, nindex) => (
          <span
            key={`Evidence-N${nindex}`}
            className={
              "flex flex-wrap justify-between items-center cursor-pointer py-1 px-3 mr-2 mb-2 rounded text-base" +
              (state.selectedReference === "N" + nindex
                ? " bg-nprimary hover:bg-nsecondary text-white"
                : " border border-nprimary text-nprimary hover:bg-nprimary hover:text-white")
            }
            onClick={() => selectReference("N" + nindex)}
          >
            N{nindex+1}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 ml-3 hover:text-gray-300"
              viewBox="0 0 20 20"
              fill="currentColor"
              onClick={() => removeReference("N" + nindex)}
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              />
            </svg>
          </span>
        ))}
      </div>
      <div
        id="actual-text"
        className="mt-5 h-full"
        onMouseUp={handleMouseUp}
        style={{ whiteSpace: "pre-wrap" }}
      >
        {state.actual_text}
      </div>
    </div>
  )
}

export default Highlighter
