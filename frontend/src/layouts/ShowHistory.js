import React, { useEffect, useState } from "react";
import { View, StyleSheet, ScrollView, SafeAreaView, Text } from "react-native";
import HistoryItem from "../components/HistoryItem";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Constants from "expo-constants";
import { useIsFocused } from "@react-navigation/native";


const STORE_HISTORY_KEY = "predictHistory";

function ShowHistory({ navigation }) {
  const [storeHistories, setStoreHistories] = useState([]);
  const isFocused = useIsFocused();

  useEffect(() => {
    if (isFocused) getDataFromStore();
  }, [isFocused]);

  const getDataFromStore = async () => {
    try {
      await AsyncStorage.getItem(STORE_HISTORY_KEY, (err, result) => {
        setStoreHistories(JSON.parse(result));
      });
    } catch (e) {
      console.log("Get Storage history error: " + e);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        {storeHistories ? storeHistories.map((item, index) => {
          return (
            <HistoryItem
              sourceURI={item.uri}
              title={item.name}
              description={item.description}
              time={item.time}
              key={index}
            />
          );
        }) : <Text>Danh sách trống</Text>}
      </ScrollView>
    </SafeAreaView>
  );
}

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

export default ShowHistory;
