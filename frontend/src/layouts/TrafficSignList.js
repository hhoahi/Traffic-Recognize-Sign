import React, { useEffect, useState } from "react";
import {
  Image,
  View,
  StyleSheet,
  ScrollView,
  SafeAreaView,
} from "react-native";
import Constants from "expo-constants";
import { useIsFocused } from "@react-navigation/native";
import { Text, ListItem, Avatar } from "react-native-elements";

import labels from "../resources/labels.json";

const LabelImages = [
  require("../../assets/images/0.png"),
  require("../../assets/images/1.png"),
  require("../../assets/images/2.png"),
  require("../../assets/images/3.png"),
  require("../../assets/images/4.png"),
  require("../../assets/images/5.png"),
  require("../../assets/images/6.png"),
  require("../../assets/images/7.png"),
  require("../../assets/images/8.png"),
  require("../../assets/images/9.png"),
  require("../../assets/images/10.png"),
  require("../../assets/images/11.png"),
  require("../../assets/images/12.png"),
  require("../../assets/images/13.png"),
  require("../../assets/images/14.png"),
  require("../../assets/images/15.png"),
  require("../../assets/images/16.png"),
  require("../../assets/images/17.png"),
  require("../../assets/images/18.png"),
  require("../../assets/images/19.png"),
  require("../../assets/images/20.png"),
  require("../../assets/images/21.png"),
  require("../../assets/images/22.png"),
  require("../../assets/images/23.png"),
  require("../../assets/images/24.png"),
  require("../../assets/images/25.png"),
  require("../../assets/images/26.png"),
  require("../../assets/images/27.png"),
  require("../../assets/images/28.png"),
  require("../../assets/images/29.png"),
  require("../../assets/images/30.png"),
  require("../../assets/images/31.png"),
  require("../../assets/images/32.png"),
  require("../../assets/images/33.png"),
  require("../../assets/images/34.png"),
  require("../../assets/images/35.png"),
  require("../../assets/images/36.png"),
  require("../../assets/images/37.png"),
  require("../../assets/images/38.png"),
  require("../../assets/images/39.png"),
  require("../../assets/images/40.png"),
  require("../../assets/images/41.png"),
  require("../../assets/images/42.png"),
];

const TrafficSignList = ({ navigation }) => {
  // console.log(labels);

  // const listImgID = () => {
  //   labels.map((item, index) => {

  //   });

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        {labels.map((item, index) => {
          return (
            <ListItem bottomDivider key={index}>
              {<Image source={LabelImages[item?._id]} />}
              <ListItem.Content>
                <ListItem.Title>{item.name}</ListItem.Title>
                <ListItem.Subtitle>{item.description}</ListItem.Subtitle>
              </ListItem.Content>
            </ListItem>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: "column",
    marginTop: Constants.statusBarHeight,
  },
  scrollView: {
    marginHorizontal: 20,
  },
});

export default TrafficSignList;
