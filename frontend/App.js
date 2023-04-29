import React from "react";
import { StyleSheet } from "react-native";
import ShowHistory from "./src/layouts/ShowHistory";
import HomeScreen from "./src/layouts/Home";
import TrafficSignList from "./src/layouts/TrafficSignList";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import Ionicons from "@expo/vector-icons/Ionicons";
import FontAwesomeIcon from "@expo/vector-icons/FontAwesome5";

const Tab = createBottomTabNavigator();

const optionsIcon = ({ route }) => ({
  tabBarIcon: ({ color, size }) => {
    let iconName;

    if (route.name === "Home") {
      iconName = "home";
    } else if (route.name === "History") {
      iconName = "history";
    } else if (route.name === 'List') {
      iconName = 'info-circle'
    }
    return <FontAwesomeIcon name={iconName} size={size} color={color} />;
  },
});

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={optionsIcon}
        tabBarOptions={{
          activeTintColor: "tomato",
          inactiveTintColor: "gray",
        }}
      >
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="History" component={ShowHistory} />
        <Tab.Screen name="List" component={TrafficSignList} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});
