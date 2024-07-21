import { PayloadAction, createSlice } from "@reduxjs/toolkit";

export interface DecodedToken {
  category: string;
  exp: number;
  iat: number;
  role: string;
  uNum: number;
  username: string;
}

// Update TokenSlice to include the decoded token
export interface TokenSlice {
  value: DecodedToken | null;
}

// Set the initial state
const initialState: TokenSlice = {
  value: null,
};

export const tokenSlice = createSlice({
  name: "token",
  initialState,
  reducers: {
    setToken: (state, action: PayloadAction<DecodedToken>) => {
      state.value = action.payload;
    },
    deleteToken: (state) => {
      state.value = null;
    },
  },
});

export const { setToken, deleteToken } = tokenSlice.actions;

export default tokenSlice.reducer;
