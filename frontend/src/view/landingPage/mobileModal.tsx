import React, { CSSProperties } from "react";

interface ModalProps {
  isVisible: boolean;
  message: string;
  onClose: () => void;
}

export const isMobileDevice = () => {
  return /Mobi|Android/i.test(navigator.userAgent);
};

export const Modal: React.FC<ModalProps> = ({
  isVisible,
  message,
  onClose,
}) => {
  if (!isVisible) return null;

  const modalStyle: CSSProperties = {
    position: "fixed",
    top: "0",
    left: "0",
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
  };

  const contentStyle: CSSProperties = {
    backgroundColor: "#fff",
    padding: "20px",
    borderRadius: "8px",
    textAlign: "center",
    whiteSpace: "pre-wrap",
  };

  const buttonStyle: CSSProperties = {
    marginTop: "20px",
  };

  return (
    <div style={modalStyle}>
      <div style={contentStyle}>
        <p>{message}</p>
      </div>
    </div>
  );
};

export default Modal;
