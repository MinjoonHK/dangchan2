import React, { useState } from "react";
import { PlusOutlined, MinusOutlined } from "@ant-design/icons";
import { Input } from "antd";

const Box = () => {
  const [boxList, setBoxList] = useState([
    {
      boxValue: "",
      iconType: "plus",
    },
  ]);

  const addBox = () => {
    setBoxList([...boxList, { boxValue: "", iconType: "minus" }]);
  };

  const removeBox = (index) => {
    setBoxList(boxList.filter((_, i) => i !== index));
  };

  return (
    <div>
      <div style={{ width: "30%" }}>
        {boxList.map((box, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            {box.iconType === "plus" ? (
              <PlusOutlined
                onClick={addBox}
                style={{ width: "25px", height: "25px" }}
              />
            ) : (
              <MinusOutlined
                onClick={() => removeBox(index)}
                style={{ width: "25px", height: "25px" }}
              />
            )}
            <Input
              style={{ marginLeft: "8px" }}
              value={box.boxValue}
              onChange={(e) => {
                const newBoxList = [...boxList];
                newBoxList[index].boxValue = e.target.value;
                setBoxList(newBoxList);
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default Box;
