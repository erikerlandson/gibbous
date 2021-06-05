name := "gibbous"

version := "0.3.1-SNAPSHOT"

//isSnapshot := true,

//publishConfiguration := publishConfiguration.value.withOverwrite(true)

publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true)

organization := "com.manyangled"

pomIncludeRepository := { _ => false }

publishMavenStyle := true

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

licenses += ("Apache-2.0", url("http://opensource.org/licenses/Apache-2.0"))

homepage := Some(url("https://github.com/isarn/isarn-sketches"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/erikerlandson/gibbous.git"),
    "scm:git@github.com:erikerlandson/gibbous.git"
  )
)

developers := List(
  Developer(
    id    = "erikerlandson",
    name  = "Erik Erlandson",
    email = "eje@redhat.com",
    url   = url("https://erikerlandson.github.io/")
  )
)

crossPaths := false // drop off Scala suffix from artifact names.

autoScalaLibrary := false // exclude scala-library from dependencies

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

// commons math used to be '% Provided' but the 'packageDoc' target
// now fails with that, so I'm just going to make it required
libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "com.novocode" % "junit-interface" % "0.11" % Test
)

compileOrder := CompileOrder.JavaThenScala

javacOptions ++= Seq()

// sbt clean xsbt unidoc; sbt previewSite; sbt ghpagesPushSite  // do clean first!

//enablePlugins(JavaUnidocPlugin, GenJavadocPlugin, PublishJavadocPlugin, GhpagesPlugin)
enablePlugins(JavaUnidocPlugin, PublishJavadocPlugin, GhpagesPlugin)

siteSubdirName in JavaUnidoc := "java/api"

addMappingsToSiteDir(mappings in (JavaUnidoc, packageDoc), siteSubdirName in JavaUnidoc)

git.remoteRepo := "git@github.com:erikerlandson/gibbous.git"
