var h4all = document.querySelectorAll("#nav h4")

gsap.to("#nav",{
    backgroundColor: "#000",
    height:"105px",
    duration:0.5,
    scrollTrigger:{
        trigger:"#nav",
        scroller:"body",
        // markers:true,
        start:"top -10%",
        end:"top -11%",
        scrub:1
    }
    
})

