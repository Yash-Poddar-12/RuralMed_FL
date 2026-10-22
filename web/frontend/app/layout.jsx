import "./globals.css";

export const metadata = {
  title: "RuralMed FL | Monitoring Dashboard",
  description: "Communication-efficient federated learning under rural connectivity constraints.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
