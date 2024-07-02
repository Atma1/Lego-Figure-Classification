import logo from "../logo512.png";

export default function Navbar() {
    return (
        <nav id="nav-bar">
            <img src={logo} className="h-8 w-8 ml-2 mt-1" alt="logo"/>
            <h1 id="web-title" className="text-2xl py-2 font-bold"><i>Lego Figure Classification</i></h1>
        </nav>
    );
}