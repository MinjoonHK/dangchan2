import React from "react";
import { DownOutlined } from "@ant-design/icons";
import type { MenuProps } from "antd";
import Dropdown from "antd/es/dropdown/dropdown";
import Space from "antd/es/space";

const items: MenuProps["items"] = [
  {
    label: <a href="https://www.antgroup.com">1st menu item</a>,
    key: "0",
  },
  {
    label: <a href="https://www.aliyun.com">2nd menu item</a>,
    key: "1",
  },
  {
    type: "divider",
  },
  {
    label: "3rd menu item",
    key: "3",
  },
];

const userDropdown: React.FC = () => (
  <Dropdown menu={{ items }} trigger={["click"]}>
    <a onClick={(e) => e.preventDefault()}>
      <Space>
        <DownOutlined />
      </Space>
    </a>
  </Dropdown>
);

export default userDropdown;
