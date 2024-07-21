import { configureStore } from "@reduxjs/toolkit";
import timerReducer from "./features/timer/timerSlice";
import userReducer from "./features/user/userSlice";
import tokenReducer from "./features/token/tokenSlice";
export const store = configureStore({
  reducer: {
    timer: timerReducer,
    user: userReducer,
    token: tokenReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
