import Layout from "./Layout"
import Form from "./pages/Form";
import Home from "./pages/Home";
import { BrowserRouter,Route,Routes } from "react-router-dom";
import Result from "./pages/Result";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout/>}>
          <Route path="/" element={<Home/>} />
          <Route path="business-form" element={<Form/>} />
          <Route path="simulation-results" element={<Result/>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;