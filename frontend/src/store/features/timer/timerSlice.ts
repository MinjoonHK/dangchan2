import { createSlice } from "@reduxjs/toolkit";

export interface TimerSlice {
  value: boolean;
}

const initialState: TimerSlice = {
  value: false,
};

export const timerSlice = createSlice({
  name: "timer",
  initialState,
  reducers: {
    start: (state) => {
      state.value = true;
    },
    stop: (state) => {
      state.value = false;
    },
  },
});

export const { start, stop } = timerSlice.actions;

export default timerSlice.reducer;
