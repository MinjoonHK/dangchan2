import { createSlice } from "@reduxjs/toolkit";

export interface UserSlice {
  value: boolean;
}

const initialState: UserSlice = {
  value: false,
};

export const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    login: (state) => {
      state.value = true;
    },
    logout: (state) => {
      state.value = false;
    },
  },
});

export const { login, logout } = userSlice.actions;

export default userSlice.reducer;
