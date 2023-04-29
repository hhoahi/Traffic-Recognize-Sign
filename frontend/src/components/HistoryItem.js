import React from "react";
import {
  StyleSheet,
  View,
  Text,
  Image,
  TouchableHighlight,
} from "react-native";
import { Card } from "react-native-elements";
import { TouchableRipple } from "react-native-paper";

function HistoryItem({ sourceURI, title, description, time }) {
  return (
    <Card>
      <TouchableRipple rippleColor="rgba(0, 0, 0, .32)">
        <View style={styles.container}>
          <Card.Image
            source={{ uri: sourceURI }}
            style={{ width: 75, height: 75 }}
          />
          <View style={styles.textDescription}>
            <Card.FeaturedTitle style={{ color: "#212121" }}>
              {title}
            </Card.FeaturedTitle>

            <Text style={{ color: "#424242", margin: 0 }}>{description}</Text>
            <Text style={{ color: "#fbc02d" }}>{time}</Text>
          </View>
        </View>
      </TouchableRipple>
    </Card>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: "row",
  },
  textDescription: {
    flex: 1,
    flexDirection: "column",
    justifyContent: "flex-start",
    left: 15,
  },
});

export default HistoryItem;
